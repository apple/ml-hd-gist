#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

# flake8: noqa

import inspect
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList, validate_stopping_criteria)
from transformers.generation.utils import (GenerationMixin,
                                           GreedySearchDecoderOnlyOutput,
                                           GreedySearchEncoderDecoderOutput,
                                           GreedySearchOutput)
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


class GistGenerationMixin(GenerationMixin):
    """Overrides GenerationMixin with special handling for gist attention masks."""

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = [
            "decoder_",
            "cross_attn",
            "use_cache",
            "cross_attention_mask",
        ]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = (
            "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        )
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value
                for argument, value in encoder_kwargs.items()
                if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        using_past_key_values = "past_key_values" in model_kwargs
        if using_past_key_values:
            warnings.warn(
                "past_key_values passed to encoder. "
                "This should only happen when reusing gist tokens."
            )
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        if using_past_key_values:
            # past_key_values should not be passed to decoder, it creates its own.
            del model_kwargs["past_key_values"]

        return model_kwargs

    def strip_slot_ids(self, key):
        # key = list(key)
        if key[0] == self.config.bos_token_id:
            key = key[1:]
        return key

    def _update_model_kwargs_for_generation(
        self,
        step,
        outputs: ModelOutput,
        input_ids: torch.tensor,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """Update model inputs, especially attention_mask_gist, for gist generation.

        EXPLANATION OF HOW ATTENTION_MASK_GIST WORKS FOR DECODER ONLY MODELS:

        Normally, when gist model is forwarded over N tokens (e.g. a prompt), we
        broadcast the attention_mask_gist to 4D: (B, 1, N, N).  This lets us
        encode the entire prompt, and lets us (1) learn the appropriate
        key/value representations for each past token, and (2) generate the
        next token (with masking correctly applied).

        For example, with N = 3 and a gist token in position 2/3, the attention
        mask might look like: (ignoring batch size and leading 1 dim)

            1 1 1
            0 1 1  <- GIST
            0 1 1

        However, in subsequent decode steps in sequential generation, we cache
        the key/value representations learned in (1). Then, this function is
        called, and the default behavior is to encode only the SINGLE new input
        id that was sampled in the previous timestep. The FIRST TIME this
        function is called (after the first token post-prompt has been
        decoded), the attention mask is the (B, 1, N, N) 4D mask described
        above.  However, in subsequent calls, we don't care about this
        attention mask anymore, since the cached key/values have already been
        computed (with masking already applied).

        Instead, since input_ids is only one token, what we need is a SINGLE
        attention_mask, of shape (B, 1, 1, N + 1), which tells the gist model
        what to attend to when decoding the next token. That is, we need

            0 1 1 1

        where you are still prevented from attending pre-gist, but the new token
        you have decoded can be attended to.

        Given either a (B, 1, N, N) or (B, 1, 1, N) attention mask, this can be
        done by keeping just the last row of the 4D attention_mask_gist, then
        adding on a 1:

            (B, 1, N, N)
            1 1 1
            0 1 1
            0 1 1

            (B, 1, 1, N)
            0 1 1

            keep last row -> (B, 1, 1, N + 1)
            0 1 1 1

        This preserves gist masking, since if there is a gist token, zero mask
        entries will persist for the rest of the decode sequence.
        """
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                assert attention_mask.ndim == 2, (
                    "Expected 2d attention mask. This code doesn't work "
                    "with 3D attention masks"
                )
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
            if "attention_mask_gist" in model_kwargs:
                attention_mask_gist = model_kwargs["attention_mask_gist"]
                assert attention_mask_gist.ndim == 4, "Expected 4D attention mask gist."
                assert attention_mask_gist.shape[1] == 1 and (
                    # Attention mask should either be (B, 1, N, N) (at the
                    # start of decoding) or (B, 1, 1, N) (midway through
                    # decoding, since input ids only considers input for
                    # the next token assuming other values are cached.
                    attention_mask_gist.shape[2] == 1
                    or attention_mask_gist.shape[2] == attention_mask_gist.shape[3]
                ), f"Got attention mask of shape {attention_mask_gist.shape}"
                # Here, M is either N or 1.
                last_row = attention_mask_gist[
                    :, :, -1:
                ]  # (B, 1, M, N) -> (B, 1, 1, N)

                bracket_id, comma_id = torch.tensor([9092]).to(input_ids.device), torch.tensor([613]).to(input_ids.device)
                '''
                Yichen Jiang: This piece of code below, before each decoding step, 
                check whether the input_ids(partially decoded outputs) ends with a categorical slot. 
                If it does, then unmask the value_gist tokens corresponding to that slot.
                '''
                if "attention_masks_gist_value" in model_kwargs and "prompt_attention_mask_gist" in model_kwargs:
                    prompt_attention_masks_gist_value = model_kwargs["attention_masks_gist_value"]
                    prompt_attention_mask_gist = model_kwargs["prompt_attention_mask_gist"]

                    seq_len = input_ids.size(-1)
                    all_last_rows = []
                    for batch_id in range(input_ids.size(0)):
                        all_last_rows.append(last_row[batch_id])
                        for _k, _v in prompt_attention_masks_gist_value[batch_id].items():
                            slot = _k.to(input_ids.device) #self.strip_slot_ids(_k).to(input_ids.device)

                            slot_len = slot.size(-1)
                            # if all(input_ids[batch_id][-slot_len:] == slot) or all(input_ids[batch_id][-(slot_len+1):-1] == slot) \
                            #     or all(input_ids[batch_id][-(slot_len+2):-2] == slot) or all(input_ids[batch_id][-(slot_len+3):-3] == slot):
                            if all(input_ids[batch_id][-slot_len:] == slot):
                                prompt_len = _v.size(-1)
                                # _v: (1, 1, len(prompt))
                                tmp_last_row = torch.cat([
                                    _v[None, None, :],  # Use the prompt_attention_masks_gist_value for this categorical arg "_k"
                                    torch.ones((1, 1, step), dtype=torch.bool).to(_v.device) # Append ones for the generated tokens
                                ], dim=-1).to(last_row.device)
                                all_last_rows[-1] = tmp_last_row
                                print(f'batch-{batch_id}, step={step}')
                                print('Zoom in to Value mask')
                                break

                            if all(input_ids[batch_id][-(slot_len+5):-5] == slot):  ## Go back to slot mask
                            # if input_ids[batch_id][-1:] == bracket_id or input_ids[batch_id][-1:] == comma_id:
                                prompt_len = prompt_attention_mask_gist.size(-1)
                                tmp_last_row = torch.cat([
                                    prompt_attention_mask_gist[batch_id, :, -1:, :],  # Use the prompt_attention_mask_gist (all value_gist is masked)
                                    torch.ones((1, 1, step), dtype=torch.bool).to(_v.device)
                                ], dim=-1).to(last_row.device)
                                all_last_rows[-1] = tmp_last_row
                                print(f'batch-{batch_id}, step={step}')
                                print('Zoom out to Slot mask')

                    last_row = torch.stack(all_last_rows, dim=0)
                    # print('new last row', last_row.size())

                attention_mask_gist = torch.cat(
                    [
                        last_row,  # (B, 1, 1, N)
                        last_row.new_ones((last_row.shape[0], 1, 1, 1)),  # (B, 1, 1, 1)
                    ],
                    dim=-1,
                )  # (B, 1, 1, N) -> (B, 1, 1, N + 1)
                model_kwargs["attention_mask_gist"] = attention_mask_gist

                # '''
                # JYC: This piece of code below is intended for the model to predict which "masks_w_valuegist" to use.
                # Instead of being a dictionary (key = categorical slot name, value = mask) like attention_masks_gist_value,
                # all_attention_masks_w_valuegist is simply a list of masks, each unmasking a different set of value_gist tokens.
                # '''
                # if "all_attention_masks_w_valuegist" in model_kwargs:
                #     all_attention_masks_w_valuegist = model_kwargs["all_attention_masks_w_valuegist"]
                #     for _ib in range(len(all_attention_masks_w_valuegist)):
                #         for slot_id in range(len(all_attention_masks_w_valuegist[_ib])):
                #             attention_masks_w_valuegist = all_attention_masks_w_valuegist[_ib][slot_id]
                #             attention_masks_w_valuegist = torch.cat(
                #                 [
                #                     attention_masks_w_valuegist,
                #                     attention_masks_w_valuegist.new_ones((1,))
                #                 ]
                #             )
                #             all_attention_masks_w_valuegist[_ib][slot_id] = attention_masks_w_valuegist
                #     model_kwargs["attention_masks_w_valuegist"] = all_attention_masks_w_valuegist

        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )

        return model_kwargs

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args and \
            key not in ["attention_masks_gist_value", "prompt_attention_masks_gist_slot", "prompt_attention_mask_gist",
                        "num_slots", "completion_input_ids", "benchmarking"]:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        completion_input_ids: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        benchmarking: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        NOTE: The only change between the huggingface greedy search and this
        function is the introduction of a "first_time" variable that doesn't
        truncate input ids when kv seqs are passed for the first time (for gist
        caching).

        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        first_time = True
        step = 0
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, first_time=first_time, **model_kwargs)
            first_time = False

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            # print(benchmarking, completion_input_ids)
            if benchmarking and completion_input_ids is not None:
                next_tokens = completion_input_ids[:, step]
                # print('appending completion input ids')
                # print(input_ids)
            #     input_ids = torch.cat([input_ids, completion_input_ids[:, step:step+1]], dim=-1)
            # else:
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                step, outputs, input_ids, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            step += 1
            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            # print(stopping_criteria(input_ids, scores))
            # print(unfinished_sequences)
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
