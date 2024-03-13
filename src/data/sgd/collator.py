#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union
import json
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

from ...utils import first_mismatch
from .. import gist

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForSGDCLM:
    """Data collator for decoder-only models. Does left padding."""

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    max_length_human: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    gist_token_id: int = 50257
    pad_token: int = 0
    add_gist_token: bool = True
    add_intent_gist_token: bool = False
    add_slot_gist_token: bool = False
    add_ctg_val_gist_token: bool = False
    intent_gist_token_id: int = 50258
    slot_gist_token_id: int = 50259
    ctg_val_gist_token_id: int = 50260
    gist_condition: str = "gist"
    num_gist_tokens: int = 10
    check_correctness: bool = False
    inbatch_reconstruct_ratio: float = 0.
    reconstruct_token_id: int = 50258
    gist_token: str = "<GIST>"
    intent_gist_token: str = "<GIST_INTENT>"
    slot_gist_token: str = "<GIST_SLOT>"
    value_gist_token: str = "<GIST_VALUE>"
    reconstruct_token: str = "<RECONSTRUCT>"

    def __post_init__(self):
        if self.max_length_human is None:
            self.max_length_human = self.max_length

    def replace_quote(self, string):
        return string.replace("\"", "\'")

    def insert_gist_tokens(self, instruction, input, completion):
        ctg_slot_value_dict = {}
        completion_str = completion

        instruct_obj = json.loads(instruction)
        instruction_str = instruct_obj['description']
        if 'intents' in instruct_obj.keys():
            all_intents = list(instruct_obj['intents'].keys())
            assert len(all_intents) == 1
            instruction_str += instruct_obj['intents'][all_intents[0]] + '. Parameters: '
        slot_values, slot_values_recon = [], []
        slot_values_dict = instruct_obj['slot_values']
        ctg_slot_rev_indices, ctg_slots = {}, []
        num_slots = len(list(slot_values_dict.keys()))
        for _is, (slot, value) in enumerate(slot_values_dict.items()):
            if 'categorical_values' in value:
                ctg_slots.append(f'"{slot}": "')
                ctg_slot_value_dict[f'"{slot}": "'] = []
                ctg_slot_rev_indices[f'"{slot}": "'] = num_slots - _is

                ctg_values, ctg_values_recon = [], []
                for v_idx, _val in value['categorical_values'].items():
                    if self.add_ctg_val_gist_token:
                        ctg_values.append(f"{v_idx}: {self.replace_quote(_val)} {self.value_gist_token}")
                    else:
                        ctg_values.append(f"{v_idx}: {self.replace_quote(_val)}")
                    ctg_values_recon.append(f"{v_idx}: {self.replace_quote(_val)}")
                    ctg_slot_value_dict[f'"{slot}": "'].append(v_idx)

                ctg_values = '{ ' + ', '.join(ctg_values) + ' } '
                ctg_values_recon = '{ ' + ', '.join(ctg_values_recon) + ' } '
                if self.add_slot_gist_token:
                    slot_values.append(f"\"{slot}\": \"{self.replace_quote(value['description'])} {ctg_values}\" {self.slot_gist_token}")
                else:
                    slot_values.append(f"\"{slot}\": \"{self.replace_quote(value['description'])} {ctg_values}\"")
                slot_values_recon.append(f"\"{slot}\": \"{self.replace_quote(value['description'])} {ctg_values_recon}\"")
            else:
                if self.add_slot_gist_token:
                    slot_values.append(f"\"{slot}\": \"{self.replace_quote(value['description'])}\" {self.slot_gist_token}")
                else:
                    slot_values.append(f"\"{slot}\": \"{self.replace_quote(value['description'])}\"")
                slot_values_recon.append(f"\"{slot}\": \"{self.replace_quote(value['description'])}\"")

        instruction_str_recon = instruction_str + ', '.join(slot_values_recon)
        instruction_str += ', '.join(slot_values)

        if input:
            prompt = f"Instruction: {instruction_str}\n{input}"  # noqa
        else:
            prompt = f"Instruction: {instruction_str}\nOutput:"  # noqa

        return prompt, instruction_str, instruction_str_recon, completion_str, ctg_slot_rev_indices, ctg_slots, ctg_slot_value_dict, num_slots
            # intent_rev_indices, intents, num_intents

    def __call__(self, batch, return_tensors=None):
        """
        Yichen Jiang: We need to make sure that no example is discarded even if it's too long. Since we need to use batch_size=1
        for 7B models and we cannot discard the only example we have.
        :param batch:
        :param return_tensors:
        :return:
        """
        if any("human" in instance["split"] for instance in batch):
            # Use the human max lengths.
            max_length = self.max_length_human
        else:
            max_length = self.max_length

        if return_tensors is None:
            return_tensors = self.return_tensors

        if any("train" in instance["split"] for instance in batch):
            is_training = True
        else:
            is_training = False

        model_inputs = defaultdict(list)
        for instance in batch:
            if not self.add_gist_token:
                # Add gist tokens later, during tokenization.
                maybe_gist_str = ""
            else:
                if self.num_gist_tokens == 0:
                    num_gist_tokens = 0

                    for i in range(20):
                        if f's{i}: ' in instance['instruction']:
                            num_gist_tokens += 1

                    maybe_gist_str = " ".join(
                        ["<GIST>" for _ in range(num_gist_tokens)]
                    )
                else:
                    maybe_gist_str = " ".join(
                        ["<GIST>" for _ in range(self.num_gist_tokens)]
                    )

            input = f"Input: {instance['input']}\nOutput:" if instance['input'] is not None else None
            input_prefix = "Input:"
            instruction = f"{instance['instruction']}"
            completion = f"{instance['output']}"
            gist_tokens = f"\n{maybe_gist_str}\n"

            add_inbatch_reconstruction = False
            if (self.inbatch_reconstruct_ratio > 0 and is_training) or self.inbatch_reconstruct_ratio == 1.0:
                rand_num = np.random.random()
                if rand_num < self.inbatch_reconstruct_ratio:
                    add_inbatch_reconstruction = True
                    input = f"Input: {instance['input']} Reconstruct the API.\nOutput:"

            num_slots, num_intents = 0, 0
            if self.add_slot_gist_token:
                prompt, instruction, instruction_recon, completion, ctg_slot_rev_indices, ctg_slots, ctg_slot_value_dict, num_slots \
                = self.insert_gist_tokens(
                    instruction,
                    input if input else None,
                    completion,
                )
            else:
                instruction_recon = instruction
                if instance["input"]:
                    prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\n{input}" \
                             f""  # noqa
                else:
                    prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nOutput:"  # noqa

            if not completion.startswith('{ '):
                completion = '{ ' + completion[1:-1] + ' }'

            if self.inbatch_reconstruct_ratio > 0 and is_training:
                # rand_num = np.random.random()
                if add_inbatch_reconstruction:
                    if self.add_slot_gist_token or self.add_ctg_val_gist_token:
                        # completion += f" {self.gist_token} {instruction.replace(f' {self.slot_gist_token}', '').replace(f' {self.value_gist_token}', '')}"
                        completion += f" {self.gist_token} {instruction_recon}"
                        # input = f"Input: {instance['input']} and reconstruct the API.\nOutput:"
                    else:
                        completion += f" {self.reconstruct_token} {instruction}"
                        # input = f"Input: {instance['input']} and reconstruct the API.\nOutput:"

            tokenized_prompt = self.tokenizer(prompt)["input_ids"]
            tokenized_completion = self.tokenizer(completion, add_special_tokens=False)[
                "input_ids"
            ] + [self.tokenizer.eos_token_id]
            tokenized_instruction = self.tokenizer(instruction)["input_ids"]
            tokenized_input = self.tokenizer(input)["input_ids"]
            tokenized_input_prefix = self.tokenizer(input_prefix)["input_ids"]
            tokenized_gist_tokens = self.tokenizer(gist_tokens)["input_ids"]

            if self.check_correctness:
                # Check that combining the prompt + completion after
                # tokenization is the same as tokenizing the prompt + completion
                # together.
                combined = tokenized_prompt + tokenized_completion
                real = self.tokenizer(prompt + " " + completion)["input_ids"] + [
                    self.tokenizer.eos_token_id
                ]
                if combined != real:
                    logger.warning(
                        (
                            "Tokenizing prompt/completion separately gave different "
                            "results. This is usually because the output is empty. "
                            "First mismatch location: %s. Source: %s",
                        ),
                        str(first_mismatch(combined, real)),
                        self.tokenizer.decode(combined),
                    )
                    continue

            tokenized_source = tokenized_instruction + tokenized_gist_tokens + tokenized_input + tokenized_completion
            if len(tokenized_source) > max_length:
                to_trim = len(tokenized_source) - max_length
                # max_prompt_len = max_length - len(tokenized_completion)
                min_input_length = 256 # min(256, int(1/3*max_prompt_len))
                # min_instruction_length = 256
                min_completion_length = 256
                #max_completion_length = max_length - min_input_length - min_instruction_length

                if len(tokenized_completion) >= min_completion_length:
                    print('trimming completion from right')
                    tokenized_completion = tokenized_completion[:-to_trim]
                elif len(tokenized_input) <= min_input_length:
                    print('trimming instruction from right')
                    print(len(tokenized_instruction), len(tokenized_input), len(tokenized_completion))
                    tokenized_instruction = tokenized_instruction[:-to_trim]
                else:
                    if to_trim <= len(tokenized_input) - min_input_length:
                        print('trimming input from left')
                        tokenized_input = tokenized_input_prefix + tokenized_input[to_trim+len(tokenized_input_prefix):]
                    else:
                        to_trim_input = len(tokenized_input) - min_input_length
                        to_trim_instruction = to_trim - to_trim_input
                        tokenized_input = tokenized_input_prefix + tokenized_input[to_trim_input+len(tokenized_input_prefix):]
                        tokenized_instruction = tokenized_instruction[:-to_trim_instruction]
                        print(f'trimming {to_trim_instruction} instruction from right and trimming {to_trim_input} input from left')
                tokenized_prompt = tokenized_instruction + tokenized_gist_tokens + tokenized_input
                assert len(tokenized_prompt) + len(tokenized_completion) == max_length, \
                    (len(tokenized_instruction), len(tokenized_gist_tokens), len(tokenized_input), len(tokenized_completion))

            tokenized_source = tokenized_prompt + tokenized_completion
            labels = [self.label_pad_token_id] * len(
                tokenized_prompt
            ) + tokenized_completion

            if self.add_slot_gist_token and self.add_ctg_val_gist_token:
                all_ctg_completion_masks = {}

                for ctg_slot in ctg_slots:
                    ctg_completion_mask = torch.zeros(len(tokenized_source))

                    for completion_idx in range(len(tokenized_completion)):
                        partial_completion = self.tokenizer.decode(tokenized_completion[:completion_idx + 1])
                        # print(partial_completion)
                        if partial_completion.endswith(ctg_slot):
                            ctg_completion_mask[completion_idx + len(tokenized_prompt)] = 1
                            if completion_idx + 3 >= len(tokenized_completion):
                                ctg_completion_mask[completion_idx + len(tokenized_prompt):] = 1
                                break

                            ctg_completion_mask[completion_idx + len(tokenized_prompt) + 1] = 1
                            ctg_completion_mask[completion_idx + len(tokenized_prompt) + 2] = 1
                            ctg_completion_mask[completion_idx + len(tokenized_prompt) + 3] = 1

                            _itok = 0
                            while True:
                                partial_completion_2 \
                                    = self.tokenizer.decode(tokenized_completion[:completion_idx + _itok + 5])
                                if partial_completion_2.endswith("\",") or partial_completion_2.endswith("}") \
                                or completion_idx + 4 + _itok == len(tokenized_completion):
                                    # or partial_completion_2.endswith("}\"") or partial_completion_2.endswith("},"):
                                    break
                                else:
                                    ctg_completion_mask[completion_idx + len(tokenized_prompt) + 4 + _itok] = 1
                                _itok += 1
                            if not self.inbatch_reconstruct_ratio > 0:
                                break

                    all_ctg_completion_masks[ctg_slot] = ctg_completion_mask

                model_inputs["all_ctg_completion_masks"].append(all_ctg_completion_masks)

            model_inputs["input_ids"].append(tokenized_source)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1 for _ in tokenized_source])

            model_inputs["prompt_input_ids"].append(tokenized_prompt)
            model_inputs["prompt_attention_mask"].append([1 for _ in tokenized_prompt])

            model_inputs["completion_input_ids"].append(tokenized_completion)
            model_inputs["completion_attention_mask"].append(
                [1 for _ in tokenized_completion]
            )
            model_inputs["uid"].append(instance['uid'])
            model_inputs["num_slots"].append(num_slots)
            if self.add_slot_gist_token:
                model_inputs["ctg_slot_rev_indices"].append(ctg_slot_rev_indices)

        # Left-pad inputs, convert to tensor.
        for key, value in model_inputs.items():
            if key == "labels":
                pad_token_id = self.label_pad_token_id
            else:
                pad_token_id = self.tokenizer.pad_token_id
            # To left-pad inputs, reverse, then right-pad, then reverse.
            if key in ['uid', 'ctg_slot_rev_indices', 'all_ctg_completion_masks', 'num_slots']:
                model_inputs[key] = value
                assert len(value) == 1, "Need to left pad all_ctg_completion_masks if bsz > 1."
            else:
                value_tensors = [torch.tensor(v[::-1]) for v in value]
                model_inputs[key] = torch.fliplr(
                    pad_sequence(
                        value_tensors,
                        batch_first=True,
                        padding_value=pad_token_id,
                    )
                )

        # Construct gist mask.
        if self.gist_condition == "gist":
            gist_fn = gist.make_gist_mask
        elif self.gist_condition == "neg_control":
            gist_fn = gist.make_neg_control_mask
        elif self.gist_condition == "pos_control":
            gist_fn = gist.make_pos_control_mask
        else:
            raise ValueError(f"Unknown gist condition {self.gist_condition}")

        if self.add_slot_gist_token and self.add_ctg_val_gist_token:
            model_inputs["attention_mask_gist"], model_inputs["prompt_attention_masks_gist_value"] = \
                gist.make_slot_value_gist_mask(
                    model_inputs["input_ids"],
                    model_inputs["prompt_input_ids"],
                    slot_gist_token=self.slot_gist_token_id,
                    value_gist_token=self.ctg_val_gist_token_id,
                    ctg_slot_rev_indices=model_inputs["ctg_slot_rev_indices"],
                    all_ctg_completion_masks=model_inputs["all_ctg_completion_masks"],
                    all_num_slots=model_inputs["num_slots"],
                    tokenizer=self.tokenizer,
                    inbatch_reconstruct=self.inbatch_reconstruct_ratio > 0 and is_training,
                    reconstruct_token=self.gist_token_id,
                )

            model_inputs["prompt_attention_mask_gist"] = gist.make_slot_gist_mask(
                model_inputs["prompt_input_ids"],
                slot_gist_token=self.slot_gist_token_id,
                all_num_slots=model_inputs["num_slots"],
                # mask_previous_slots=self.mask_previous_slots,
            )

        elif self.add_slot_gist_token:
            model_inputs["attention_mask_gist"] = gist.make_slot_gist_mask(
                model_inputs["input_ids"],
                slot_gist_token=self.slot_gist_token_id,
                all_num_slots=model_inputs["num_slots"],
                # mask_previous_slots=self.mask_previous_slots,
            )
            model_inputs["prompt_attention_mask_gist"] = gist.make_slot_gist_mask(
                model_inputs["prompt_input_ids"],
                slot_gist_token=self.slot_gist_token_id,
                all_num_slots=model_inputs["num_slots"],
                # mask_previous_slots=self.mask_previous_slots,
            )
        else:
            model_inputs["attention_mask_gist"] = gist_fn(
                model_inputs["input_ids"],
                self.gist_token_id,
                inbatch_reconstruct=self.inbatch_reconstruct_ratio > 0 and is_training,
                reconstruct_token=self.reconstruct_token_id,
            )
            model_inputs["prompt_attention_mask_gist"] = gist_fn(
                model_inputs["prompt_input_ids"],
                self.gist_token_id,
            )

        return model_inputs
