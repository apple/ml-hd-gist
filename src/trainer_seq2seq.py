#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.testing import assert_close
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig, Seq2SeqTrainer
from transformers.debug_utils import DebugOption
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.utils import is_torch_tpu_available, logging

from .benchmarking import profile
from .data.gist import get_first_pad_index, get_gist_index
from .data.utils import strip_special_tokens
from .utils import first_mismatch, MyEvalPrediction
from .gist_llama import GistLlamaForCausalLM
from .gist_t5 import GistT5ForConditionalGeneration

logger = logging.get_logger(__name__)


if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm  # type: ignore
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl  # type: ignore

from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_apex_available
)
from packaging import version


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_apex_available():
    from apex import amp

from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback

from collections.abc import Mapping


def find_batch_size_with_list(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    if isinstance(tensors, (list, tuple)):
        # for t in tensors:
        #     result = find_batch_size(t)
        #     if result is not None:
        #         return result
        return len(tensors)
    elif isinstance(tensors, Mapping):
        for key, value in tensors.items():
            result = find_batch_size_with_list(value)
            if result is not None:
                return result
    elif isinstance(tensors, torch.Tensor):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None
    elif isinstance(tensors, np.ndarray):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None

class GistSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            tag = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            update_gist_token_only=False,
            gist_token=None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics
        )
        self.tag = tag
        self.update_gist_token_only = update_gist_token_only
        self.gist_token = gist_token

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, defaultdict):
            return {k: self._prepare_input(v) for k, v in data.items()}
        elif isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.deepspeed and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.args.hf_deepspeed_config.dtype()})
            return data.to(**kwargs)
        return data

    def select_attention_mask(self, attention_mask, all_batch_attention_mask, completion_mask, ids, num_slots, use_posterior=False):
        for _ib, all_attention_mask in enumerate(all_batch_attention_mask):
            for slot_id in range(num_slots[_ib]):
                # print(ids[_ib])

                new_completion_mask = torch.eq(ids[_ib], torch.tensor([slot_id]).to(attention_mask.device))
                # print(new_completion_mask)
                if not use_posterior:
                    new_completion_mask = torch.cat(
                        [torch.zeros_like(new_completion_mask[0:1]).to(new_completion_mask.device),
                         new_completion_mask[:-1]]
                    )
                #     print(new_completion_mask)
                # print('\n')

                new_completion_mask = new_completion_mask & completion_mask[_ib]
                # print(new_completion_mask)
                attention_mask[_ib] = torch.where(
                    new_completion_mask[None, :, None].type(torch.bool),
                    all_attention_mask[slot_id][None, None, :],
                    attention_mask[_ib]
                )
        return attention_mask

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
            # print('label smoothed loss')
            # print(loss)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # print('no label smooth loss')
            # print(loss)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        if self.update_gist_token_only:
            if isinstance(self.model, GistT5ForConditionalGeneration):
                if self.model.shared.weight.grad is not None:
                    self.model.shared.weight.grad[:self.gist_token, :] = torch.zeros_like(
                        self.model.shared.weight.grad[:self.gist_token, :]
                    )
                    self.model.shared.weight.grad[self.gist_token+1:, :] = torch.zeros_like(
                        self.model.shared.weight.grad[self.gist_token+1:, :]
                    )
            elif isinstance(self.model, GistLlamaForCausalLM):
                self.model.shared.weight.grad[:self.gist_token, :] = torch.zeros_like(
                    self.model.shared.weight.grad[:self.gist_token, :]
                )
                self.model.shared.weight.grad[self.gist_token + 1:, :] = torch.zeros_like(
                    self.model.shared.weight.grad[self.gist_token + 1:, :]
                )
        return loss.detach()

    def skip_non_gist_examples(
        self,
        inputs: Dict[str, torch.Tensor],
        gist_token: int,
        input_ids_key: str = "input_ids",
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Skip examples that do not contain the gist token.

        Returns None if all examples are skipped.
        """
        has_gist = (inputs[input_ids_key] == gist_token).any(-1)
        if not has_gist.any():
            return None
        ### JYC: We added non-tensor inputs so we cannot just use indexes like [True, True, False] to select examples with GIST
        if has_gist.all():
            return inputs
        else:
            raise NotImplementedError
            return {k: v[has_gist] for k, v in inputs.items()}

    def remove_gist(
        self,
        inputs: Dict[str, torch.Tensor],
        num_gist_tokens: int,
        gist_indices: Optional[Tuple[int]] = None,
        gist_token: Optional[int] = None,
        benchmarking_prompt_caching=False
    ) -> Dict[str, torch.Tensor]:
        """
        Given inputs to an HF model, return the same inputs but with gist tokens
        removed.

        Adjusts the attention mask to assume cached gist tokens.

        This method is used to prepare a batch for inference with cached gist tokens.
        """
        gist_inputs_to_repad = defaultdict(list)

        is_encoder_decoder = hasattr(self.model, "encoder")
        # T5.
        for i, input_row in enumerate(inputs["input_ids"]):
            if gist_indices is None:
                if gist_token is None:
                    raise ValueError("Must provide either gist_indices or gist_token.")
                gist_start, _ = get_gist_index(
                    input_row, gist_token, raise_if_no_tokens=True
                )
            else:
                if gist_token is not None:
                    logger.warning(
                        "Both gist_indices and gist_token were provided. "
                        "Using gist_indices."
                    )
                gist_start = gist_indices[i]

            if is_encoder_decoder:
                # Compute padding start, since inputs are padded to the right.
                pad_start = get_first_pad_index(input_row, self.tokenizer.pad_token_id)
            else:
                # Otherwise, we're in a decoder-only setting, and inputs are
                # padded to the left.
                pad_start = len(input_row)

            input_ids_length = None
            for k, v in inputs.items():
                if k == "completion_input_ids":
                    gist_inputs_to_repad[k].append(v[i])
                elif k == "input_ids":
                    input_ids_post_gist = v[i, gist_start + num_gist_tokens : pad_start]
                    # The below assertion could fail if truncation with no eos token.
                    if is_encoder_decoder:  # End of encoder inputs should be EOS
                        assert input_ids_post_gist[-1] == self.tokenizer.eos_token_id
                    gist_inputs_to_repad[k].append(input_ids_post_gist)
                    input_ids_length = len(input_ids_post_gist)
                elif k == "attention_mask":
                    # FIXME(jayelm): it's possible I don't even need any of this
                    # fancy attention mask truncation, I can just leave the args
                    # empty.

                    # We don't need any special gist masking here. Just
                    # retrieve the relevant last row of the attention
                    # matrix, including the gist tokens, and trim.
                    if benchmarking_prompt_caching:
                        if is_encoder_decoder:
                            attention_mask_post_gist = v[i, -1, gist_start+1:pad_start]
                        else:
                            attention_mask_post_gist = v[i, gist_start+1:pad_start]
                    else:
                        if is_encoder_decoder:
                            attention_mask_post_gist = v[i, -1, gist_start:pad_start]
                        else:
                            attention_mask_post_gist = v[i, gist_start:pad_start]
                    if not (attention_mask_post_gist == 1).all():
                        raise ValueError(
                            "Attention mask is not all ones. Are you trying to "
                            "cache gists in the negative control setting?"
                        )
                    assert (attention_mask_post_gist == 1).all()
                    gist_inputs_to_repad[k].append(attention_mask_post_gist)
                elif k == "cross_attention_mask":
                    cross_attention_mask_post_gist = v[i, gist_start:pad_start]
                    if not (cross_attention_mask_post_gist == 1).all():
                        raise ValueError(
                            "Cross attention mask is not all ones. Are you trying to "
                            "cache gists in the negative control setting?"
                        )
                    gist_inputs_to_repad[k].append(cross_attention_mask_post_gist)
                elif k == "attention_mask_gist":
                    if benchmarking_prompt_caching:
                        attention_mask_gist_post_gist = v[i, -1, -1, gist_start+1:pad_start]
                    else:
                        attention_mask_gist_post_gist = v[i, -1, -1, gist_start:pad_start]
                    # NOTE: This assertion is not necessarily true if we are
                    # caching the entire prompt.
                    # assert (attention_mask_gist_post_gist == 1).all()
                    gist_inputs_to_repad[k].append(attention_mask_gist_post_gist)
                else:
                    # We don't need to repad decoder inputs.
                    pass
            # Assert all of the items we added are the same length.
            if benchmarking_prompt_caching:
                assert all(
                    len(v[-1]) == (input_ids_length + num_gist_tokens - 1)
                    for k, v in gist_inputs_to_repad.items()
                    if k != "input_ids" and k != "completion_input_ids"
                )
            else:
                assert all(
                    len(v[-1]) == (input_ids_length + num_gist_tokens)
                    for k, v in gist_inputs_to_repad.items()
                    if k != "input_ids" and k != "completion_input_ids"
                )

        gist_inputs = {}
        for k, vs in gist_inputs_to_repad.items():
            if k == "input_ids":
                pad_value = self.tokenizer.pad_token_id
            else:
                pad_value = 0

            if not is_encoder_decoder:
                # Right pad inputs.
                vs = [torch.flip(v, (0,)) for v in vs]

            padded_vs = pad_sequence(vs, batch_first=True, padding_value=pad_value)

            if not is_encoder_decoder:
                # Flip back.
                padded_vs = torch.fliplr(padded_vs)

            gist_inputs[k] = padded_vs

        if not is_encoder_decoder:
            # Attention mask gist should be 4D.
            gist_inputs["attention_mask_gist"] = gist_inputs["attention_mask_gist"][
                :, None, None
            ]
        return gist_inputs

    def remove_slot_value_gist(
        self,
        inputs: Dict[str, torch.Tensor],
        gist_indices: Optional[Tuple[int]] = None,
        gist_token: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Given inputs to an HF model, return the same inputs but with gist tokens
        removed.

        Adjusts the attention mask to assume cached gist tokens.

        This method is used to prepare a batch for inference with cached gist tokens.
        """
        gist_inputs_to_repad = defaultdict(list)

        is_encoder_decoder = hasattr(self.model, "encoder")
        # T5.
        gist_offsets = []
        all_num_gist_tokens = []
        for i, input_row in enumerate(inputs["input_ids"]):
            if gist_indices is None:
                raise NotImplementedError
                if gist_token is None:
                    raise ValueError("Must provide either gist_indices or gist_token.")
                gist_start, _ = get_gist_index(
                    input_row, gist_token, raise_if_no_tokens=True
                )
            else:
                if gist_token is not None:
                    logger.warning(
                        "Both gist_indices and gist_token were provided. "
                        "Using gist_indices."
                    )
                gist_start = gist_indices[i][0]
                gist_end = gist_indices[i][-1]
                num_gist_tokens = len(gist_indices[i])
                all_num_gist_tokens.append(num_gist_tokens)
                gist_offsets.append(gist_end - num_gist_tokens + 1)
            if is_encoder_decoder:
                # Compute padding start, since inputs are padded to the right.
                pad_start = get_first_pad_index(input_row, self.tokenizer.pad_token_id)
            else:
                # Otherwise, we're in a decoder-only setting, and inputs are
                # padded to the left.
                pad_start = len(input_row)

            input_ids_length = None
            for k, v in inputs.items():
                if k == "completion_input_ids":
                    gist_inputs_to_repad[k].append(v[i])
                elif k == "input_ids":
                    input_ids_post_gist = v[i, gist_end + 1 : pad_start]
                    # The below assertion could fail if truncation with no eos token.
                    if is_encoder_decoder:  # End of encoder inputs should be EOS
                        assert input_ids_post_gist[-1] == self.tokenizer.eos_token_id
                    gist_inputs_to_repad[k].append(input_ids_post_gist)
                    input_ids_length = len(input_ids_post_gist)
                    # print(k)
                    # print(input_ids_post_gist.size())
                elif k == "attention_mask":
                    # FIXME(jayelm): it's possible I don't even need any of this
                    # fancy attention mask truncation, I can just leave the args
                    # empty.

                    # We don't need any special gist masking here. Just
                    # retrieve the relevant last row of the attention
                    # matrix, including the gist tokens, and trim.
                    if is_encoder_decoder:
                        raise NotImplementedError
                        attention_mask_post_gist = v[i, -1, gist_start:pad_start]
                    else:
                        attention_mask_post_gist = v[i, gist_end + 1 - num_gist_tokens : pad_start]
                    if not (attention_mask_post_gist == 1).all():
                        raise ValueError(
                            "Attention mask is not all ones. Are you trying to "
                            "cache gists in the negative control setting?"
                        )
                    assert (attention_mask_post_gist == 1).all()
                    # print(k)
                    # print(attention_mask_post_gist.size())
                    gist_inputs_to_repad[k].append(attention_mask_post_gist)
                elif k == "cross_attention_mask":
                    raise NotImplementedError ## JYC: We do not consider encoder-decoder models for dynamic gisting
                    cross_attention_mask_post_gist = v[i, gist_start:pad_start]
                    if not (cross_attention_mask_post_gist == 1).all():
                        raise ValueError(
                            "Cross attention mask is not all ones. Are you trying to "
                            "cache gists in the negative control setting?"
                        )
                    gist_inputs_to_repad[k].append(cross_attention_mask_post_gist)
                elif k == "attention_mask_gist":
                    # print(k)
                    attention_mask_gist_post_gist_part1 = v[i, -1, -1, gist_indices[i]]
                    attention_mask_gist_post_gist_part2 = v[i, -1, -1, gist_end+1:pad_start]
                    attention_mask_gist_post_gist = torch.cat(
                        [attention_mask_gist_post_gist_part1, attention_mask_gist_post_gist_part2],
                        dim=-1
                    )
                    # print(attention_mask_gist_post_gist.size())
                    # NOTE: This assertion is not necessarily true if we are
                    # caching the entire prompt.
                    # assert (attention_mask_gist_post_gist == 1).all()
                    gist_inputs_to_repad[k].append(attention_mask_gist_post_gist)
                elif k == "prompt_attention_masks_gist_value":
                    attention_masks_gist_value_to_repad = {}
                    for slot_k, slot_v in v[i].items():
                        # print(slot_v)
                        attention_mask_gist_post_gist_part1 = slot_v[gist_indices[i]]
                        attention_mask_gist_post_gist_part2 = slot_v[gist_end+1:pad_start]
                        attention_mask_gist_post_gist = torch.cat(
                            [attention_mask_gist_post_gist_part1, attention_mask_gist_post_gist_part2],
                            dim=-1
                        )
                        attention_masks_gist_value_to_repad[slot_k] = attention_mask_gist_post_gist
                        # print(attention_mask_gist_post_gist)
                        # print('\n')
                        # print(attention_mask_gist_post_gist.size())
                    # NOTE: This assertion is not necessarily true if we are
                    # caching the entire prompt.
                    # assert (attention_mask_gist_post_gist == 1).all()
                    gist_inputs_to_repad[k].append(attention_masks_gist_value_to_repad)
                else:
                    # We don't need to repad decoder inputs.
                    pass
            # Assert all of the items we added are the same length.
            assert all(
                len(v[-1]) == (input_ids_length + num_gist_tokens)
                for k, v in gist_inputs_to_repad.items()
                if k != "input_ids" and k != "completion_input_ids" and k != "prompt_attention_masks_gist_value"
            )

        '''
        JYC: No need to repad since we are only dealing with batch_size = 1
        '''
        gist_inputs = {}
        for k, vs in gist_inputs_to_repad.items():
        #     if k == "input_ids":
        #         pad_value = self.tokenizer.pad_token_id
        #     else:
        #         pad_value = 0
        #
        #     if not is_encoder_decoder:
        #         # Right pad inputs.
        #         vs = [torch.flip(v, (0,)) for v in vs]
        #
        #     padded_vs = pad_sequence(vs, batch_first=True, padding_value=pad_value)
        #
        #     if not is_encoder_decoder:
        #         # Flip back.
        #         padded_vs = torch.fliplr(padded_vs)
            if k == "prompt_attention_masks_gist_value":
                padded_vs = vs
            else:
                padded_vs = torch.stack(vs, dim=0)

            gist_inputs[k] = padded_vs

        if not is_encoder_decoder:
            # Attention mask gist should be 4D.
            gist_inputs["attention_mask_gist"] = gist_inputs["attention_mask_gist"][:, None, None]
            gist_inputs["prompt_attention_mask_gist"] = gist_inputs["attention_mask_gist"]
        return gist_inputs, torch.stack(gist_offsets, dim=0), all_num_gist_tokens


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        benchmarking = False
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model.
                Most models expect the targets under the argument `labels`.
                Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor],
            Optional[torch.Tensor]]: A tuple with the loss, logits and labels
            (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        is_encoder_decoder = hasattr(self.model, "encoder")
        original_inputs = None
        if not is_encoder_decoder and "prompt_input_ids" in inputs:
            # Decoder-only models: when generating, we don't want to generate
            # with the full text, but with the prompt text.
            original_inputs = {
                k: inputs[k]
                for k in ["input_ids", "attention_mask", "attention_mask_gist"]
            }
            inputs["input_ids"] = inputs["prompt_input_ids"]
            inputs["attention_mask"] = inputs["prompt_attention_mask"]
            inputs["attention_mask_gist"] = inputs["prompt_attention_mask_gist"]

            if "prompt_all_attention_masks_w_valuegist" in inputs:
                original_inputs["all_attention_masks_w_valuegist"] = inputs["all_attention_masks_w_valuegist"]
                original_inputs["slot_gist_only_mask"] = inputs["slot_gist_only_mask"]
                inputs["all_attention_masks_w_valuegist"] = inputs["prompt_all_attention_masks_w_valuegist"]
                inputs["slot_gist_only_mask"] = inputs["prompt_slot_gist_only_mask"]

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        # ==== BEGIN GIST CHANGES ====
        if "attention_mask_gist" in inputs:
            gen_kwargs["attention_mask_gist"] = inputs.get("attention_mask_gist", None)
        if "cross_attention_mask" in inputs:
            gen_kwargs["cross_attention_mask"] = inputs.get(
                "cross_attention_mask", None
            )
        if "gist_activations" in inputs:
            gen_kwargs["gist_activations"] = inputs.get("gist_activations", None)
        if "past_key_values" in inputs:
            gen_kwargs["past_key_values"] = inputs.get("past_key_values", None)
        if "gist_offset" in inputs:
            gen_kwargs["gist_offset"] = inputs.get("gist_offset", None)
        if "prompt_attention_masks_gist_value" in inputs:
            gen_kwargs["attention_masks_gist_value"] = inputs.get("prompt_attention_masks_gist_value", None)
        if "prompt_attention_mask_gist" in inputs:
            '''
            JYC: We need to keep a second copy of this "prompt_attention_mask_gist" other than "attention_mask_gist".
            Because in _update_model_kwargs_for_generation, we keep updating the attention_mask_gist but we need
            the original "prompt_attention_mask_gist"
            '''
            gen_kwargs["prompt_attention_mask_gist"] = inputs.get("prompt_attention_mask_gist", None)
        if "prompt_all_attention_masks_w_valuegist" in inputs:
            gen_kwargs["all_attention_masks_w_valuegist"] = inputs.get("prompt_all_attention_masks_w_valuegist", None)
        if "slot_gist_only_mask" in inputs:
            gen_kwargs["slot_gist_only_mask"] = inputs.get("slot_gist_only_mask", None)
        if "num_slots" in inputs:
            gen_kwargs["num_slots"] = inputs.get("num_slots", None)
        if benchmarking and "completion_input_ids" in inputs:
            gen_kwargs["benchmarking"] = True
            gen_kwargs["completion_input_ids"] = inputs.get("completion_input_ids", None)

        # ==== END GIST CHANGES ====
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None
            )

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if (
            is_encoder_decoder
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        if not is_encoder_decoder:
            # FIXME(jayelm): this hack makes llama work only. There's some
            # deepspeed bug where the generation config contains
            # pad token -1
            # eos token [1]
            # max length None
            # When it should be
            # bos token 1, eos token 2, pad token 0.
            # This also means beam search, etc. won't work (greedy decode
            # only).
            logger.warning(
                "Overwriting existing generation config due to "
                "DeepSpeed bug. If model is not LLAMA, check this."
            )
            gen_kwargs["generation_config"] = GenerationConfig(
                # max_length=512,
                do_sample=False,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        if not is_encoder_decoder:
            # The prompt is included in the generated tokens. Remove this.
            assert (
                generated_tokens[:, : generation_inputs.shape[-1]] == generation_inputs
            ).all()
            generated_tokens = generated_tokens[:, generation_inputs.shape[-1] :]

        # in case the batch is shorter than max length, the output should be padded
        if (
            gen_kwargs.get("max_length") is not None
            and generated_tokens.shape[-1] < gen_kwargs["max_length"]
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"]
            )
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[
            -1
        ] < (gen_kwargs["max_new_tokens"] + 1):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_new_tokens"] + 1
            )

        # Replace original inputs when computing standard LM loss.
        if not is_encoder_decoder and original_inputs is not None:
            inputs.update(original_inputs)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if (
                gen_kwargs.get("max_length") is not None
                and labels.shape[-1] < gen_kwargs["max_length"]
            ):
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, (gen_kwargs["max_new_tokens"] + 1)
                )
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        resume_from_checkpoint: Optional[str] = None,
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute
        metrics, as they are task-dependent (pass it to the init
        `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If
                it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must
                implement the `__len__` method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a
                dictionary) that should be ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For
                example the metrics "bleu" will be named "eval_bleu" if the
                prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential
            metrics computed from the predictions. The dictionary also contains
            the epoch number which comes from the training state.
        """
        gen_kwargs = gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            if self.args.max_new_tokens is not None:
                gen_kwargs["max_new_tokens"] = self.args.max_new_tokens
            else:
                gen_kwargs["max_length"] = self.args.generation_max_length

        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics,
            # otherwise we defer to self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile,
            # execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    # def _nested_gather_string(self, non_tensors, name=None):
    #     """
    #     Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    #     concatenating them to `gathered`
    #     """
    #     if non_tensors is None:
    #         return
    #     if is_torch_tpu_available():
    #         raise NotImplementedError
    #     elif is_sagemaker_mp_enabled():
    #         raise NotImplementedError
    #     elif (self.args.distributed_state is not None and self.args.distributed_state.distributed_type != "NO") or (
    #             self.args.distributed_state is None and self.args.local_rank != -1
    #     ):
    #         output_tensors = [tensor.clone() for _ in range(torch.disttributed.get_world_size())]
    #         torch.disttributed.all_gather(output_tensors, tensor)
    #         concat = torch.cat(output_tensors, dim=0)
    #     return non_tensors


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        resume_from_checkpoint: Optional[str] = None,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and
        `Trainer.predict()`.  Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should
            # be able to do eval from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self,
                num_training_steps=1_000_000,
                resume_from_checkpoint=resume_from_checkpoint,
                inference=resume_from_checkpoint is None,
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or
        # ``predict`` isn't called while ``train`` is running, cast it to the
        # right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(
                args.device
            )

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        uid_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_uids = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader
                # in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
            uid = inputs['uid']
            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            inputs_decode = (
                self._prepare_input(inputs["input_ids"])
                if args.include_inputs_for_metrics
                else None
            )

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            # if uid is not None:
            #     uid_host = (
            #         uid if uid_host is None
            #         else nested_concat(uid_host, uid)
            #     )

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done
            # enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(all_preds, logits, padding_index=-100)
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(
                            all_inputs, inputs_decode, padding_index=-100
                        )
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=-100)
                    )
                # if uid_host is not None:
                #     all_uids = (
                #         uid_host if all_uids is None
                #         else nested_concat(all_uids, uid_host)
                #     )
                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )
        # if uid_host is not None:
        #     all_uids = (
        #         uid_host if all_uids is None
        #         else nested_concat(all_uids, uid_host)
        #     )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type,
        # but whether the dataset has the right methods. Therefore we need to
        # make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a
        # distributed training, the number of samplers has been rounded to a
        # multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            if args.write_outputs:
                if args.gist.add_slot_gist_token:
                    if args.gist.add_ctg_val_gist_token:
                        output_fname = f"outputs-{self.state.global_step}-{eval_dataset._split}-{self.tag}-dyna_gist-seed{args.seed}.csv"
                    else:
                        output_fname = f"outputs-{self.state.global_step}-{eval_dataset._split}-{self.tag}-slot_gist-seed{args.seed}.csv"
                else:
                    output_fname = f"outputs-{self.state.global_step}-{eval_dataset._split}-{self.tag}-gist{args.gist.num_gist_tokens}-seed{args.seed}.csv"
                output_file = os.path.join(
                    args.output_dir,
                    output_fname, #f"outputs-{self.state.global_step}-{eval_dataset._split}.csv",
                )
            else:
                output_file = None
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds, label_ids=all_labels, inputs=all_inputs
                    ),
                    output_file=output_file,
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels),
                    output_file=output_file,
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[
                f"{metric_key_prefix}_jit_compilation_time"
            ] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    @torch.no_grad()
    def benchmark(
        self,
        gist_token: int,
        slot_gist_token: Optional[int] = None,
        value_gist_token: Optional[int] = None,
        eval_dataset: Optional[Dataset] = None,
        output_file: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run benchmarking and returns metrics.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If
                it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must
                implement the `__len__` method.
        Returns:
            A dictionary containing the benchmarks.
        """
        gen_kwargs = gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=False,
        )

        df = self.benchmark_loop(
            eval_dataloader,
            gist_token,
            slot_gist_token,
            value_gist_token,
            output_file=output_file,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        return df

    def benchmark_loop(
        self,
        dataloader: DataLoader,
        gist_token: int,
        slot_gist_token: Optional[int] = None,
        value_gist_token: Optional[int] = None,
        output_file: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Benchmark loop. Tips:

        training.do_benchmarking_sanity_checks=True checks that input is the
        same with and without gist caching.

        For llama (decoder-only) models, set benchmarking_prompt_caching=True to
        compare prompt caching vs no prompt caching (vs caching gist tokens
        only).

        If you want 3 way comparison between standard decoding, gist decoding,
        and prompt caching decoding, you need to run this script twice, once
        with benchmarking_prompt_caching=True and once with it False.

        gist.condition=pos_control will also automatically cache prompts (rather
        than gist activations). This can be useful for benchmarking sanity
        checks. But if you want to compare FLOPs, then the model needs to
        produce the same output, and you need to use the same model. So use a
        gist model with gist.condition=gist and prompt caching on or off.
        """
        args = self.args

        if output_file is None:
            output_file = args.benchmarking_output_file

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should
            # be able to do eval from the checkpoint eventually
            with torch.set_grad_enabled(True):
                # This has to be evaluated with grad enabled, even though
                # benchmark has grads turned off.
                deepspeed_engine, _, _ = deepspeed_init(
                    self,
                    num_training_steps=1_000_000,
                    resume_from_checkpoint=resume_from_checkpoint,
                    inference=resume_from_checkpoint is None,
                )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or
        # ``predict`` isn't called while ``train`` is running, cast it to the
        # right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info("***** Benchmarking *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(
                args.device
            )

        if args.past_index >= 0:
            self._past = None

        observed_num_examples = 0

        is_encoder_decoder = hasattr(self.model, "encoder")

        profiler_records = []

        # Profile the prediction step.
        def forward_to_profile(model, *args, **kwargs):
            return model(*args, **kwargs)

        def generate_to_profile(model, *args, **kwargs):
            return self.prediction_step(model, *args, **kwargs)

        if args.benchmarking_generation_of_groundtruth:
            profiled_function = profile(
                generate_to_profile, profiler_type=self.args.benchmarking_profiler
            )
        else:
            profiled_function = profile(
                forward_to_profile, profiler_type=self.args.benchmarking_profiler
            )

        if self.args.max_benchmarking_samples is not None and self.args.max_benchmarking_samples > 0:
            max_benchmarking_samples = self.args.max_benchmarking_samples
        else:
            max_benchmarking_samples = len(dataloader)

        # Main evaluation loop
        pbar = tqdm(
            zip(dataloader, range(max_benchmarking_samples)),
            total=max_benchmarking_samples,
        )
        for step, (inputs, _) in enumerate(pbar):
            # Skip examples with no gist tokens, if it happens.
            if args.gist.add_slot_gist_token:
                inputs = self.skip_non_gist_examples(inputs, slot_gist_token)
                if inputs is None:
                    logger.warning("Batch had no slot gist tokens, skipping.")
                    continue
            else:
                inputs = self.skip_non_gist_examples(inputs, gist_token)
                if inputs is None:
                    logger.warning("Batch had no gist tokens, skipping.")
                    continue

            # Update the observed num examples
            observed_batch_size = find_batch_size_with_list(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader
                # in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            orig_labels = inputs["labels"]
            labels_length = (orig_labels != -100).sum().item()

            inputs = self._prepare_inputs(inputs)

            # Cache the gist activations.
            # Remove extra inputs for each model, and cache the gist activations.
            if is_encoder_decoder:
                # T5
                inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "cross_attention_mask": inputs["cross_attention_mask"],
                }
                gist_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
                if args.gist.condition == "pos_control":
                    raise RuntimeError(
                        "You should not be benchmarking pos_control for T5 since "
                        "we can't cache prompts"
                    )
            else:
                if observed_batch_size != 1:
                    raise ValueError(
                        "Only batch size 1 is supported for now, due to rotary offsets."
                    )
                if args.gist.add_slot_gist_token and args.gist.add_ctg_val_gist_token:
                    if args.benchmarking_generation_of_groundtruth:
                        inputs = {
                            "completion_input_ids": inputs["completion_input_ids"],
                            "input_ids": inputs["prompt_input_ids"],
                            "attention_mask": inputs["prompt_attention_mask"],
                            "attention_mask_gist": inputs["prompt_attention_mask_gist"],
                            "prompt_attention_masks_gist_value": inputs["prompt_attention_masks_gist_value"]
                        }
                    else:
                        inputs = {
                            "input_ids": inputs["prompt_input_ids"],
                            "attention_mask": inputs["prompt_attention_mask"],
                            "attention_mask_gist": inputs["prompt_attention_mask_gist"],
                            "prompt_attention_masks_gist_value": inputs["prompt_attention_masks_gist_value"]
                        }
                else:
                    if args.benchmarking_generation_of_groundtruth:
                        inputs = {
                            "completion_input_ids": inputs["completion_input_ids"],
                            "input_ids": inputs["prompt_input_ids"],
                            "attention_mask": inputs["prompt_attention_mask"],
                            "attention_mask_gist": inputs["prompt_attention_mask_gist"],
                        }
                    else:
                        inputs = {
                            "input_ids": inputs["prompt_input_ids"],
                            "attention_mask": inputs["prompt_attention_mask"],
                            "attention_mask_gist": inputs["prompt_attention_mask_gist"],
                        }

                gist_kwargs = inputs

            if args.gist.condition == "pos_control" or args.benchmarking_prompt_caching:
                assert args.gist.num_gist_tokens == 1
                # Decoder-only models can also cache the prompt (i.e. everything
                # up to the gist), which we should also benchmark.
                # args.gist.condition=pos_control is useful when paired with
                # do_benchmarking_sanity_checks to verify that prompt caching
                # does indeed work.

                # In this case, we're actually caching all activations, not just
                # gist activations. This is akin to caching N gist tokens #
                # where N = prompt_length + original num gist tokens, hence why
                # we define effective_num_gist_tokens.
                assert (
                    inputs["input_ids"].shape[0] == 1
                ), "Can only cache all if batch size is 1 for now."
                gist_start, _ = get_gist_index(
                    inputs["input_ids"][0], gist_token, raise_if_no_tokens=True
                )
                effective_num_gist_tokens = gist_start + args.gist.num_gist_tokens
                gist_activations = model.get_gist_activations(
                    gist_token=gist_token,
                    num_gist_tokens=effective_num_gist_tokens,
                    cache_all=True,
                    **gist_kwargs,
                )
            else:
                if args.gist.add_slot_gist_token:# and args.gist.add_ctg_val_gist_token:
                    gist_activations = model.get_slot_value_gist_activations(
                        slot_gist_token=slot_gist_token,
                        value_gist_token=value_gist_token,
                        **gist_kwargs,
                    )
                else:
                    effective_num_gist_tokens = args.gist.num_gist_tokens
                    gist_activations = model.get_gist_activations(
                        gist_token=gist_token,
                        num_gist_tokens=effective_num_gist_tokens,
                        **gist_kwargs,
                    )
            if args.gist.add_slot_gist_token: # and args.gist.add_ctg_val_gist_token:
                gist_inputs, gist_offsets, all_num_gist_tokens = self.remove_slot_value_gist(
                    inputs,
                    gist_indices=gist_activations.gist_indices,
                )

                assert len(all_num_gist_tokens) == 1, "We cannot deal with batch_size > 1."
                effective_num_gist_tokens = all_num_gist_tokens[0]
            else:
                # Convert inputs, removing everything pre-gist.
                gist_inputs = self.remove_gist(
                    inputs,
                    num_gist_tokens=effective_num_gist_tokens,
                    gist_indices=gist_activations.gist_indices,
                    benchmarking_prompt_caching=args.benchmarking_prompt_caching
                )
            if is_encoder_decoder:
                # Set gist_activations
                gist_inputs["gist_activations"] = gist_activations
            else:
                # Just set past key value and gist offset, since less gist logic
                # is required.
                gist_inputs["past_key_values"] = gist_activations.past_key_values
                if args.gist.add_slot_gist_token: # and args.gist.add_ctg_val_gist_token:
                    gist_inputs["gist_offset"] = gist_offsets
                else:
                    gist_inputs["gist_offset"] = gist_activations.gist_indices

            if args.do_benchmarking_sanity_checks:
                self.benchmarking_sanity_checks(
                    model,
                    inputs,
                    gist_inputs,
                    gist_activations,
                    effective_num_gist_tokens=effective_num_gist_tokens,
                )

            if is_encoder_decoder:
                inputs["decoder_input_ids"] = torch.ones(
                    (1, 1), dtype=torch.long, device=inputs["input_ids"].device
                )
                gist_inputs["decoder_input_ids"] = torch.ones(
                    (1, 1), dtype=torch.long, device=inputs["input_ids"].device
                )
            if args.benchmarking_prompt_caching:
                inputs = {
                    "completion_input_ids": inputs["completion_input_ids"],
                    "input_ids": inputs["input_ids"][:, 1:],
                    "attention_mask": inputs["attention_mask"][:, 1:],
                    "attention_mask_gist": inputs["attention_mask_gist"][:, :, 1:, 1:],
                }

            # Randomize order of evaluation.
            if self.args.benchmarking_generation_of_groundtruth:
                if np.random.rand() < 0.5:
                    first_run = "standard"
                    if args.benchmarking_prompt_caching:
                        _, profiler_outputs, standard_prof_table = profiled_function(
                            model,
                            inputs,
                            prediction_loss_only=False,
                            benchmarking=True
                        )
                        time.sleep(1)
                    _, gist_profiler_outputs, gist_prof_table = profiled_function(
                        model,
                        gist_inputs,
                        prediction_loss_only=False,
                        benchmarking=True
                    )
                else:
                    first_run = "gist"
                    _, gist_profiler_outputs, gist_prof_table = profiled_function(
                        model,
                        gist_inputs,
                        prediction_loss_only=False,
                        benchmarking=True
                    )
                    if args.benchmarking_prompt_caching:
                        time.sleep(1)
                        _, profiler_outputs, standard_prof_table = profiled_function(
                            model,
                            inputs,
                            prediction_loss_only=False,
                            benchmarking=True
                        )
            else:
                if np.random.rand() < 0.5:
                    first_run = "standard"
                    if args.benchmarking_prompt_caching:
                        _, profiler_outputs, standard_prof_table = profiled_function(
                            model,
                            **inputs,
                        )
                        time.sleep(1)
                    _, gist_profiler_outputs, gist_prof_table = profiled_function(
                        model,
                        **gist_inputs,
                    )
                else:
                    first_run = "gist"
                    _, gist_profiler_outputs, gist_prof_table = profiled_function(
                        model,
                        **gist_inputs,
                    )
                    if args.benchmarking_prompt_caching:
                        time.sleep(1)
                        _, profiler_outputs, standard_prof_table = profiled_function(
                            model,
                            **inputs,
                        )
            # print(gist_inputs['input_ids'].size())
            # print(gist_prof_table)
            # exit()
            if args.gist.add_slot_gist_token: # and args.gist.add_ctg_val_gist_token:
                gist_index_record = gist_activations.gist_indices.detach().cpu().numpy()
            else:
                gist_index_record = gist_activations.gist_indices.item()

            record_info = {
                "id": step,
                "num_prompt_tokens": len(inputs["input_ids"][0]),
                "num_input_tokens": len(gist_inputs["input_ids"][0]),
                "effective_num_gist_tokens": effective_num_gist_tokens,
                "gist_index": gist_index_record,
                "first_run": first_run,
            }

            if args.benchmarking_prompt_caching:
                profiler_records.extend(
                    [
                        {
                            **record_info,
                            "model": "standard",
                            **profiler_outputs.to_json(),
                        },
                        {
                            **record_info,
                            "model": "gist",
                            **gist_profiler_outputs.to_json(),
                        },
                    ]
                )
            else:
                profiler_records.extend(
                    [
                        {
                            **record_info,
                            "model": "gist",
                            **gist_profiler_outputs.to_json(),
                        },
                    ]
                )

            pd.DataFrame(profiler_records).to_csv(output_file, index=False)

        return pd.DataFrame(profiler_records)

    @torch.no_grad()
    def benchmarking_sanity_checks(
        self,
        model,
        inputs,
        gist_inputs,
        gist_activations,
        effective_num_gist_tokens: Optional[int] = None,
    ):
        """
        Effective num gist tokens is there because pos control modifies the
        number of gist tokens.
        """
        if effective_num_gist_tokens is None:
            effective_num_gist_tokens = self.args.gist.num_gist_tokens
        is_encoder_decoder = hasattr(model, "encoder")

        if is_encoder_decoder:
            forward_cls = model.encoder
            encoder_inputs = {
                k: v for k, v in inputs.items() if k != "cross_attention_mask"
            }
            gist_encoder_inputs = {
                k: v for k, v in gist_inputs.items() if k != "cross_attention_mask"
            }
        else:
            forward_cls = model.model
            encoder_inputs = inputs
            gist_encoder_inputs = gist_inputs

        outputs = forward_cls(
            **encoder_inputs,
            output_hidden_states=True,
            use_cache=True,
        )

        gist_outputs = forward_cls(
            **gist_encoder_inputs,
            output_hidden_states=True,
            use_cache=True,
        )

        reps_to_check = (
            # hidden states
            ("hidden_states", outputs.hidden_states, gist_outputs.hidden_states, None),
            # last hidden state
            (
                "last_hidden_state",
                [outputs.last_hidden_state],
                [gist_outputs.last_hidden_state],
                [gist_activations.last_hidden_state],
            ),
            # keys
            (
                "keys",
                list(kv[0] for kv in outputs.past_key_values),
                list(kv[0] for kv in gist_outputs.past_key_values),
                list(kv[0] for kv in gist_activations.past_key_values),
            ),
            # values
            (
                "values",
                list(kv[1] for kv in outputs.past_key_values),
                list(kv[1] for kv in gist_outputs.past_key_values),
                list(kv[1] for kv in gist_activations.past_key_values),
            ),
        )

        for batch_idx, gist_idx in enumerate(gist_activations.gist_indices):
            # Assert that the hidden states are the same: `layer` after
            # gist token, and gist_layer from the start.
            if is_encoder_decoder:
                pad_start = get_first_pad_index(
                    inputs["input_ids"][batch_idx], self.tokenizer.pad_token_id
                )
                gist_pad_start = get_first_pad_index(
                    gist_inputs["input_ids"][batch_idx], self.tokenizer.pad_token_id
                )
                assert (
                    pad_start - gist_idx == gist_pad_start + effective_num_gist_tokens
                ), "padding does not match"
            else:
                pad_start = len(inputs["input_ids"][batch_idx])
                gist_pad_start = len(gist_inputs["input_ids"][batch_idx])

            for name, reps, gist_reps, orig_gist_reps in reps_to_check:
                if name == "hidden_states":
                    # Check reps after gist token.
                    rep_slice = slice(gist_idx + effective_num_gist_tokens, pad_start)
                    gist_rep_slice = slice(0, gist_pad_start)
                    assert orig_gist_reps is None
                elif name == "last_hidden_state" and not is_encoder_decoder:
                    # Seems like LLaMA models do not include the gist tokens in
                    # the last hidden state.
                    # Therefore it also doesn't make sense to compare to the
                    # gist activations since they aren't included.
                    rep_slice = slice(gist_idx + effective_num_gist_tokens, pad_start)
                    gist_rep_slice = slice(0, gist_pad_start)
                    # Don't compare to original gist reps, since the last hidden
                    # state does not exist in the gist encoder outputs.
                    orig_gist_reps = None
                else:
                    # Check reps including gist tokens.
                    rep_slice = slice(gist_idx, pad_start)
                    gist_rep_slice = slice(
                        0, gist_pad_start + effective_num_gist_tokens
                    )
                    assert orig_gist_reps is not None

                for layer_i, (rep, gist_rep) in enumerate(zip(reps, gist_reps)):
                    # Assert that the hidden states are the same with their
                    # respective slices.
                    rep_to_check = rep[batch_idx, ..., rep_slice, :]
                    gist_rep_to_check = gist_rep[batch_idx, ..., gist_rep_slice, :]
                    try:
                        assert_close(
                            rep_to_check, gist_rep_to_check, atol=1e-2, rtol=1e-2
                        )
                        if orig_gist_reps is not None:
                            # Check that the first slice of both reps are equal
                            # to the cached gist activations.
                            assert_close(
                                rep_to_check[..., :effective_num_gist_tokens, :],
                                orig_gist_reps[layer_i][batch_idx],
                                atol=1e-2,
                                rtol=1e-2,
                            )
                            assert_close(
                                gist_rep_to_check[..., :effective_num_gist_tokens, :],
                                orig_gist_reps[layer_i][batch_idx],
                                atol=1e-2,
                                rtol=1e-2,
                            )
                    except AssertionError as e:
                        logger.warning(
                            "Failed sanity check for batch element %d %s layer %d",
                            batch_idx,
                            name,
                            layer_i,
                        )
                        logger.warning(
                            "rep %s, gist rep %s", rep_to_check, gist_rep_to_check
                        )
                        logger.warning(
                            "first mismatch: %s",
                            first_mismatch(
                                rep_to_check.cpu().numpy().flatten(),
                                gist_rep_to_check.cpu().numpy().flatten(),
                            ),
                        )
                        logger.warning(
                            "Input ids: %s, gist input ids: %s",
                            inputs["input_ids"][batch_idx, rep_slice],
                            gist_inputs["input_ids"][batch_idx, gist_rep_slice],
                        )
                        logger.warning(
                            "(note: mismatch might be in a separate assertion)"
                        )
                        raise e

            # Verify prediction step is also the same
            _, logits, _ = self.prediction_step(model, inputs, False, ignore_keys=None)
            _, gist_logits, _ = self.prediction_step(
                model, gist_inputs, False, ignore_keys=None
            )
            input_text = list(
                map(
                    strip_special_tokens,
                    self.tokenizer.batch_decode(inputs["input_ids"]),
                )
            )
            preds = self.tokenizer.batch_decode(logits, skip_special_tokens=True)
            gist_input_text = list(
                map(
                    strip_special_tokens,
                    self.tokenizer.batch_decode(gist_inputs["input_ids"]),
                )
            )
            gist_preds = self.tokenizer.batch_decode(
                gist_logits, skip_special_tokens=True
            )

            logger.info("Outputs: %s", preds)
            logger.info("Gist outputs: %s", gist_preds)
            # Note generated sequences might differ due to length.
            # Also note this assertion might fail for large models and fp16/bf16
            # due to numerical instability.
            assert [
                p.startswith(g) or g.startswith(p) for p, g in zip(preds, gist_preds)
            ], first_mismatch(preds, gist_preds)
