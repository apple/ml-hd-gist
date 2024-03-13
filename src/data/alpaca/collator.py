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
class DataCollatorForAlpaca:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    max_source_length_human: Optional[int] = None
    max_target_length_human: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    gist_token: int = 32100
    pad_token: int = 0
    add_gist_token: bool = True
    add_intent_gist_token: bool = False
    add_slot_gist_token: bool = False
    add_ctg_val_gist_token: bool = False
    intent_gist_token: int = 32101
    slot_gist_token: int = 32102
    ctg_val_gist_token: int = 32103
    gist_condition: str = "gist"
    num_gist_tokens: int = 10

    def __post_init__(self):
        if self.max_source_length_human is None:
            self.max_source_length_human = self.max_source_length
        if self.max_target_length_human is None:
            self.max_target_length_human = self.max_target_length

    def __call__(self, batch, return_tensors=None):
        if any("human" in instance["split"] for instance in batch):
            # Use the human max lengths.
            max_source_length = self.max_source_length_human
            max_target_length = self.max_target_length_human
        else:
            max_source_length = self.max_source_length
            max_target_length = self.max_target_length

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            if not self.add_gist_token:
                # Add gist tokens later, during tokenization.
                maybe_gist_str = ""
            else:
                maybe_gist_str = "<GIST>" * self.num_gist_tokens

            if instance["input"]:
                source = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nInput: {instance['input']}"  # noqa
            else:
                # No input, instruction only.
                source = f"Instruction: {instance['instruction']}\n{maybe_gist_str}"

            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= max_source_length:
                tokenized_source = tokenized_source[:-1]  # Drop the </s> token.
            else:
                tokenized_source = tokenized_source[:max_source_length]
            sources.append(self.tokenizer.decode(tokenized_source))

        model_inputs = self.tokenizer(
            sources,
            max_length=max_source_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Tokenize labels.
        labels = [instance["output"] for instance in batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                labels,
                max_length=max_target_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        label_mask = labels["attention_mask"].bool()
        model_inputs["labels"] = labels["input_ids"].masked_fill(
            ~label_mask, self.label_pad_token_id
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=model_inputs["labels"]
            )
            model_inputs["decoder_input_ids"] = decoder_input_ids

        # modify attention mask
        if self.gist_condition == "pos_control" or not self.add_gist_token:
            # Don't change anything, just set cross attention mask.
            model_inputs["cross_attention_mask"] = model_inputs["attention_mask"]
        elif self.gist_condition == "gist":
            model_inputs["attention_mask"] = gist.make_gist_mask(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            ).squeeze(
                1
            )  # Squeeze dim 1 as T5 codebase expects 3D mask.
            # Decoder cross attn cannot see prior to the first gist token.
            model_inputs["cross_attention_mask"] = gist.make_mask_pre_first_gist(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            )
        elif self.gist_condition == "neg_control":
            model_inputs["attention_mask"] = gist.make_neg_control_mask(
                model_inputs["input_ids"],
                self.gist_token,
                pad_token=self.pad_token,
            ).squeeze(
                1
            )  # Squeeze dim 1 as T5 codebase expects 3D mask.
            # Decoder cross attn cannot see prior to (and including) *any* gist
            # token.
            model_inputs["cross_attention_mask"] = 1 - (
                gist.make_mask_post_last_gist(
                    model_inputs["input_ids"],
                    self.gist_token,
                    pad_token=self.pad_token,
                )
            )
        else:
            raise ValueError(f"Invalid gist_condition: {self.gist_condition}")

        return model_inputs


@dataclass
class DataCollatorForAlpacaCLM:
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
    predict_intent: bool = False
    intent_gist_token_id: int = 50258
    slot_gist_token_id: int = 50259
    ctg_val_gist_token_id: int = 50260
    gist_condition: str = "gist"
    num_gist_tokens: int = 10
    check_correctness: bool = False
    mask_previous_slots: bool = False
    add_chat_gist_tokens: bool = False
    user_gist_token_id: int = 50261
    system_gist_token_id: int = 50262
    end_of_chat_token_id: int = 50263
    multi_api: bool = False
    mask_previous_intents: bool = False
    inbatch_reconstruct_ratio: float = 0.
    reconstruct_token_id: int = 50258
    gist_token: str = "<GIST>"
    intent_gist_token: str = "<GIST_INTENT>"
    slot_gist_token: str = "<GIST_SLOT>"
    value_gist_token: str = "<GIST_VALUE>"
    user_gist_token: str = "<GIST_USER>"
    system_gist_token: str = "<GIST_SYSTEM>"
    end_of_chat_token: str = "<END_CHAT>"
    reconstruct_token: str = "<RECONSTRUCT>"
    detailed_completion_ratio: float = 0.
    predict_mask: bool = False
    add_slot_desc_at_the_end: bool = False

    def __post_init__(self):
        if self.max_length_human is None:
            self.max_length_human = self.max_length

    def replace_quote(self, string):
        return string.replace("\"", "\'")

    def insert_gist_tokens(self, instruction, input, completion, replacein_gist=True):
        # intent_gist_token = "<GIST_INTENT>"
        # slot_gist_token = "<GIST_SLOT>"
        # value_gist_token = "<GIST_VALUE>"
        # user_gist_token = "<GIST_USER>"
        # system_gist_token = "<GIST_SYSTEM>"
        # end_of_chat_token = "<END_CHAT>"
        ctg_slot_rev_indices, ctg_slots = {}, []
        ctg_slot_value_dict = {}
        intent_rev_indices, intents, num_intents = {}, [], 0
        completion_str = completion
        if self.multi_api:
            assert self.predict_intent
            instruct_objs = json.loads(instruction)
            all_instruction_strs = []
            all_intent_strs = []

            num_slots = sum(map(len, [list(instruct_obj['slot_values'].keys()) for instruct_obj in instruct_objs]))
            cum_num_slots = 0
            num_intents = len(instruct_objs)
            if replacein_gist:
                for _iintent, instruct_obj in enumerate(instruct_objs):
                    new_intent = f"{instruct_obj['intent'][1:]} {intent_gist_token}"
                    instruction_str = f"{instruct_obj['description']}: {new_intent}: "
                    all_intent_strs.append(
                        f"{instruct_obj['description']}: {new_intent}")
                    intents.append(f'"intent": "{new_intent}",')
                    intent_rev_indices[f'"intent": "{new_intent}",'] = num_intents - _iintent
                    slot_values = []
                    slot_values_dict = instruct_obj['slot_values']

                    for _is, (slot, value) in enumerate(slot_values_dict.items()):
                        new_slot = f"{new_intent} {slot.split('.')[-1][1:]} {slot_gist_token}"
                        if 'categorical_values' in value:
                            ctg_slots.append(f'"{new_slot}": "')
                            ctg_slot_rev_indices[f'"{new_slot}": "'] = num_slots - _is - cum_num_slots
                            ctg_values = []
                            for v_idx, _val in value['categorical_values'].items():
                                v_idx = v_idx.split(".")[-1]
                                if self.add_ctg_val_gist_token:
                                    ctg_values.append(f"{_val}: {v_idx} {value_gist_token}")
                                else:
                                    ctg_values.append(f"{_val}: {v_idx}")
                            ctg_values = '(' + ', '.join(ctg_values) + ')'
                            if self.add_slot_gist_token:
                                slot_values.append(f"{value['description']} {ctg_values}: {slot.split('.')[-1][1:]} {slot_gist_token}")
                            else:
                                slot_values.append(f"{value['description']} {ctg_values}: {slot.split('.')[-1][1:]}")
                        else:
                            if self.add_slot_gist_token:
                                slot_values.append(f"{value['description']}: {slot.split('.')[-1][1:]} {slot_gist_token}")
                            else:
                                slot_values.append(f"{value['description']}: {slot.split('.')[-1][1:]}")
                    cum_num_slots += len(list(slot_values_dict.keys()))
                    instruction_str += '{' + ', '.join(slot_values) + '}'
                    # if self.add_intent_gist_token:
                    #     # instruction_str += ' ' + f"intent {instruct_obj['intent']}: {instruct_obj['description']}. " + intent_gist_token
                    #     instruction_str += ' ' + intent_gist_token
                    all_instruction_strs.append(instruction_str)
                # instruction_str = ' '.join(all_intent_strs) + ' ' + ' '.join(all_instruction_strs)
                instruction_str = ' '.join(all_instruction_strs)

                completion_obj = json.loads(completion)
                new_completion_obj = {}
                for _param_key, _param_value in completion_obj.items():
                    if _param_key == 'intent':
                        new_completion_obj['intent'] = f"{_param_value[1:]} {intent_gist_token}"
                    else:
                        intent_idx, param_idx = _param_key.split('.')
                        intent_idx, param_idx = intent_idx[1:], param_idx[1:]
                        _new_param_key = f"{intent_idx} {intent_gist_token} {param_idx} {slot_gist_token}"
                        print(f'"{_new_param_key}": "')
                        if f'"{_new_param_key}": "' in ctg_slots:
                            value_idx = _param_value.split('.')[-1]
                            _new_param_value = f"{intent_idx} {intent_gist_token} {param_idx} {slot_gist_token} {value_idx} {value_gist_token}"
                            new_completion_obj[_new_param_key] = _new_param_value
                        else:
                            new_completion_obj[_new_param_key] = _param_value
                completion_str = json.dumps(new_completion_obj)
                # print(instruction_str)
                # print(new_completion_obj)
                # print(ctg_slot_rev_indices)
                # print(ctg_slots)
                # print(num_slots)
                # print(intent_rev_indices)
                # print(intents)
                # print(completion_str)
                # exit()

            else:
                for _iintent, instruct_obj in enumerate(instruct_objs):
                    instruction_str = f"intent {instruct_obj['intent']}: {instruct_obj['description']}. "
                    all_intent_strs.append(f"intent {instruct_obj['intent']}: {instruct_obj['description']}. {intent_gist_token}")
                    intents.append(f'"intent": "{instruct_obj["intent"]}",')
                    intent_rev_indices[f'"intent": "{instruct_obj["intent"]}",'] = num_intents - _iintent
                    slot_values = []
                    slot_values_dict = instruct_obj['slot_values']

                    for _is, (slot, value) in enumerate(slot_values_dict.items()):
                        if 'categorical_values' in value:
                            ctg_slots.append(f'"{slot}": "')
                            ctg_slot_rev_indices[f'"{slot}": "'] = num_slots - _is - cum_num_slots
                            # ctg_slots.append(f'{slot}: ')
                            # ctg_slot_rev_indices[f'{slot}: '] = num_slots - _is - cum_num_slots
                            ctg_values = []
                            for v_idx, _val in value['categorical_values'].items():
                                if self.add_ctg_val_gist_token:
                                    ctg_values.append(f"{v_idx}: {_val} {value_gist_token}")
                                else:
                                    ctg_values.append(f"{v_idx}: {_val}")
                            ctg_values = '{' + ', '.join(ctg_values) + '}'
                            if self.add_slot_gist_token:
                                slot_values.append(f"{slot}: {value['description']} {ctg_values} {slot_gist_token}")
                            else:
                                slot_values.append(f"{slot}: {value['description']} {ctg_values}")
                        else:
                            if self.add_slot_gist_token:
                                slot_values.append(f"{slot}: {value['description']} {slot_gist_token}")
                            else:
                                slot_values.append(f"{slot}: {value['description']}")

                    cum_num_slots += len(list(slot_values_dict.keys()))
                    instruction_str += ', '.join(slot_values)
                    if self.add_intent_gist_token:
                        # instruction_str += ' ' + f"intent {instruct_obj['intent']}: {instruct_obj['description']}. " + intent_gist_token
                        instruction_str += ' ' + intent_gist_token
                    all_instruction_strs.append(instruction_str)
                instruction_str = ' '.join(all_intent_strs) + ' ' + ' '.join(all_instruction_strs)
            # print(instruction_str)
            # print(ctg_slots)
            # print(ctg_slot_rev_indices)
            # print(num_slots)
            # exit()
        else:
            instruct_obj = json.loads(instruction)
            instruction_str = instruct_obj['description']
            if not self.predict_intent and 'intents' in instruct_obj.keys():
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
                    # if self.inbatch_reconstruct_ratio > 0:
                    #     ctg_slots.append(f'{slot}:')
                    #     ctg_slot_value_dict[f'{slot}:'] = []
                    #     ctg_slot_rev_indices[f'{slot}:'] = num_slots - _is

                    ctg_values, ctg_values_recon = [], []
                    for v_idx, _val in value['categorical_values'].items():
                        if self.add_ctg_val_gist_token:
                            ctg_values.append(f"{v_idx}: {self.replace_quote(_val)} {self.value_gist_token}")
                        else:
                            ctg_values.append(f"{v_idx}: {self.replace_quote(_val)}")
                        ctg_values_recon.append(f"{v_idx}: {self.replace_quote(_val)}")
                        ctg_slot_value_dict[f'"{slot}": "'].append(v_idx)
                        # if self.inbatch_reconstruct_ratio > 0:
                        #     ctg_slot_value_dict[f'{slot}:'].append(v_idx)

                    ctg_values = '{ ' + ', '.join(ctg_values) + ' } '
                    ctg_values_recon = '{ ' + ', '.join(ctg_values_recon) + ' } '
                    if self.add_slot_gist_token:
                        if self.add_slot_desc_at_the_end:
                            slot_values.append(
                                f"\"{slot}\": \"{self.replace_quote(value['description'])} {ctg_values}\" {slot}: {self.replace_quote(value['description'])} {self.slot_gist_token}")
                        else:
                            slot_values.append(f"\"{slot}\": \"{self.replace_quote(value['description'])} {ctg_values}\" {self.slot_gist_token}")
                        # slot_values_recon.append(
                        #     f"\"{slot}\": \"{value['description']} {ctg_values_recon}\"")
                    else:
                        slot_values.append(f"\"{slot}\": \"{self.replace_quote(value['description'])} {ctg_values}\"")
                    slot_values_recon.append(f"\"{slot}\": \"{self.replace_quote(value['description'])} {ctg_values_recon}\"")
                else:
                    if self.add_slot_gist_token:
                        slot_values.append(f"\"{slot}\": \"{self.replace_quote(value['description'])}\" {self.slot_gist_token}")
                        # slot_values_recon.append(f"\"{slot}\": \"{value['description']}\"")
                    else:
                        slot_values.append(f"\"{slot}\": \"{self.replace_quote(value['description'])}\"")
                    slot_values_recon.append(f"\"{slot}\": \"{self.replace_quote(value['description'])}\"")

            instruction_str_recon = instruction_str + ', '.join(slot_values_recon)
            instruction_str += ', '.join(slot_values)

            # print(instruction_str)
            # exit()
        if input:
            if self.add_chat_gist_tokens:
                input_sentences = input.split("USER:")
                # print(input_sentences)
                input = "\"USER:" + (self.system_gist_token + " USER:").join(input_sentences[1:])
                # print(input)
                input_sentences = input.split("SYSTEM:")
                # print(input_sentences)
                input = (self.user_gist_token + " SYSTEM:").join(input_sentences) + " " + self.end_of_chat_token

            prompt = f"Instruction: {instruction_str}\n{input}"  # noqa
        else:
            prompt = f"Instruction: {instruction_str}\nOutput:"  # noqa

        # if not completion_str.startswith('{ '):
        #     completion_str = '{ ' + completion_str[1:-1] + ' }'
        return prompt, instruction_str, instruction_str_recon, completion_str, ctg_slot_rev_indices, ctg_slots, ctg_slot_value_dict, num_slots, \
            intent_rev_indices, intents, num_intents

    def __call__(self, batch, return_tensors=None):
        """
        JYC: We need to make sure that no example is discarded even if it's too long. Since we need to use batch_size=1
        for 6B models and we cannot discard the only example we have.
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

        #: 9092: "}    613: ",
        # print(self.tokenizer("\"}"))
        # print(self.tokenizer("\""))
        # print(self.tokenizer("{\"s0\": \"detective\"}"))
        # print(self.tokenizer.decode([573, 9092]))
        # print(self.tokenizer.decode([9092]))
        # print(self.tokenizer.decode([376, 29913]))
        # print(self.tokenizer.decode([29913]))
        #
        # print(self.tokenizer("\","))
        # print(self.tokenizer("{\"s0\": \"detective\","))
        # print(self.tokenizer("{\"s0\": \"detective\", "))
        # print(self.tokenizer("{\"s0\": \"detective\", \"s5\""))
        # print(self.tokenizer.decode([613]))
        # print(self.tokenizer.decode([613, 29871]))
        # exit()
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

            use_detailed_completion = False
            if self.detailed_completion_ratio > 0:
                rand_num = np.random.random()
                if rand_num < self.detailed_completion_ratio:
                    input = f"Input: {instance['input']} Explain the slot and values.\nOutput:"
                    use_detailed_completion = True

            add_inbatch_reconstruction = False
            if (self.inbatch_reconstruct_ratio > 0 and is_training) or self.inbatch_reconstruct_ratio == 1.0:
                rand_num = np.random.random()
                if rand_num < self.inbatch_reconstruct_ratio:
                    add_inbatch_reconstruction = True
                    # if self.add_slot_gist_token or self.add_ctg_val_gist_token:
                    if use_detailed_completion:
                        input = f"Input: {instance['input']} Explain the slot and values and reconstruct the API.\nOutput:"
                    else:
                        input = f"Input: {instance['input']} Reconstruct the API.\nOutput:"
                    # else:
                    #     input = f"Input: {instance['input']} Reconstruct the API.\nOutput:"

            num_slots, num_intents = 0, 0
            if self.add_intent_gist_token or self.add_slot_gist_token:
                prompt, instruction, instruction_recon, completion, ctg_slot_rev_indices, ctg_slots, ctg_slot_value_dict, num_slots, intent_rev_indices, intents, num_intents \
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

            if self.detailed_completion_ratio > 0:
                completion = json.dumps(json.loads(instance['output'])['output'])
                if use_detailed_completion:
                    detailed_output = json.loads(instance['output'])['detailed_output']
                    detailed_completion = {}
                    for _slot in detailed_output:
                        detailed_slot = f"{_slot} ( {detailed_output[_slot]['description']} )"
                        if isinstance(detailed_output[_slot]['value'], str):
                            detailed_value = detailed_output[_slot]['value']
                        else:
                            _ctg_value = list(detailed_output[_slot]['value'].keys())[0]
                            detailed_value = f"{_ctg_value} ( {detailed_output[_slot]['value'][_ctg_value]} )"
                        detailed_completion[detailed_slot] = detailed_value
                    completion += f'; {json.dumps(detailed_completion)}'

                    # input = f"Input: {instance['input']} and explain the slot and values.\nOutput:"
                # else:
                #     completion = json.dumps(json.loads(instance['output'])['output'])

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
            # print(input)
            # print(instruction)
            # print(completion)
            # print('\n')

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

            # if len(tokenized_instruction)

            # tokenized_source = tokenized_prompt + tokenized_completion
            # labels = [self.label_pad_token_id] * len(
            #     tokenized_prompt
            # ) + tokenized_completion

            tokenized_source = tokenized_instruction + tokenized_gist_tokens + tokenized_input + tokenized_completion
            if len(tokenized_source) > max_length:
                to_trim = len(tokenized_source) - max_length
                max_prompt_len = max_length - len(tokenized_completion)
                # ideal_max_instruction_length = int(max_source_length * 3/4)
                min_input_length = 256 # min(256, int(1/3*max_prompt_len))
                min_instruction_length = 256
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
                # print(max_length)
                assert len(tokenized_prompt) + len(tokenized_completion) == max_length, \
                    (len(tokenized_instruction), len(tokenized_gist_tokens), len(tokenized_input), len(tokenized_completion))

            tokenized_source = tokenized_prompt + tokenized_completion
            labels = [self.label_pad_token_id] * len(
                tokenized_prompt
            ) + tokenized_completion
            # print(self.tokenizer("\"2 <GIST_INTENT> 3 <GIST_SLOT>\""))
            # print(self.tokenizer("\"i0.s2.3\""))
            # print(self.tokenizer.decode(self.tokenizer("\"i0.s2.3\"")["input_ids"]))
            # print(self.tokenizer.decode(self.tokenizer("\"2 <GIST_INTENT> 3 <GIST_SLOT>\"")["input_ids"]))
            # exit()

            if (not self.predict_mask) and self.add_slot_gist_token and self.add_ctg_val_gist_token:
                all_ctg_completion_masks = {}
                # print(completion)

                for ctg_slot in ctg_slots:
                    # print('\n')
                    # print(instance['input'])
                    # print(ctg_slot)
                    ctg_completion_mask = torch.zeros(len(tokenized_source))
                    unmask_ctg_gist_in_all_steps = False
                    if completion in ctg_slot_value_dict[ctg_slot]:
                        unmask_ctg_gist_in_all_steps = True
                        assert False

                    for ctg_value in ctg_slot_value_dict[ctg_slot]:
                        if instance['input'].endswith(ctg_value):
                            unmask_ctg_gist_in_all_steps = True
                            assert False

                    if unmask_ctg_gist_in_all_steps:
                        ctg_completion_mask[len(tokenized_prompt):] = 1
                    else:
                        for completion_idx in range(len(tokenized_completion)):
                            partial_completion = self.tokenizer.decode(tokenized_completion[:completion_idx + 1])
                            # print(partial_completion)
                            if partial_completion.endswith(ctg_slot):
                                # print(ctg_slot)
                                # print(partial_completion)

                                # print(self.tokenizer.decode(tokenized_completion[completion_idx - 10 : completion_idx + 1 + 7]))
                                # ctg_completion_mask[completion_idx + len(tokenized_prompt) - 2] = 1
                                # ctg_completion_mask[completion_idx + len(tokenized_prompt) - 1] = 1
                                # ctg_completion_mask[completion_idx + len(tokenized_prompt)] = 1
                                # ctg_completion_mask[completion_idx + len(tokenized_prompt) + 1] = 1
                                # ctg_completion_mask[completion_idx + len(tokenized_prompt) + 2] = 1
                                # ctg_completion_mask[completion_idx + len(tokenized_prompt) + 3] = 1
                                if self.multi_api:
                                    ctg_completion_mask[completion_idx + len(tokenized_prompt)] = 1
                                    ctg_completion_mask[completion_idx + len(tokenized_prompt) + 1] = 1
                                    ctg_completion_mask[completion_idx + len(tokenized_prompt) + 2] = 1
                                    for _itok in range(6):
                                        partial_completion_2 \
                                            = self.tokenizer.decode(tokenized_completion[:completion_idx + _itok + 5])
                                        if partial_completion_2.endswith("\",") or partial_completion_2.endswith("\"}"):
                                            break
                                        else:
                                            ctg_completion_mask[completion_idx + len(tokenized_prompt) + 3 + _itok] = 1

                                    # print(ctg_completion_mask)
                                    # ctg_completion_mask[completion_idx + len(tokenized_prompt) + 4] = 1
                                    # ctg_completion_mask[completion_idx + len(tokenized_prompt) + 5] = 1
                                    # ctg_completion_mask[completion_idx + len(tokenized_prompt) + 6] = 1
                                    # partial_completion_2 = self.tokenizer.decode(tokenized_completion[:completion_idx + 9])
                                    # if not partial_completion_2.endswith("\",") and not partial_completion_2.endswith("\"}"):
                                    #     ctg_completion_mask[completion_idx + len(tokenized_prompt) + 7] = 1
                                        # print(partial_completion_2)
                                        # print(ctg_slot)
                                else:
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

                                    # print(_itok)
                                # print(partial_completion)
                                # print(self.tokenizer.tokenize(': "s6.1"'))
                                # print(self.tokenizer(': "s6.1"'))
                                # print(self.tokenizer.tokenize('"s6.1"'))
                                # print(self.tokenizer('"s6.1"'))

                                if not self.inbatch_reconstruct_ratio > 0:
                                    break
                    # print(ctg_slot)
                    # print(ctg_completion_mask)
                    all_ctg_completion_masks[ctg_slot] = ctg_completion_mask

                model_inputs["all_ctg_completion_masks"].append(all_ctg_completion_masks)
                if self.add_intent_gist_token:
                    gt_intent = json.loads(completion)['intent']
                    gt_intent = f'"intent": "{gt_intent}",'
                    all_intent_completion_masks = {}

                    intent_completion_mask, completion_mask = torch.zeros(len(tokenized_source)), np.zeros(len(tokenized_source))
                    for intent in intents:
                        all_intent_completion_masks[intent] = intent_completion_mask
                        # No need to calculate completion mask for distractor intents

                    for completion_idx in range(len(tokenized_completion)):
                        partial_completion = self.tokenizer.decode(tokenized_completion[:completion_idx + 1])
                        if partial_completion.endswith(gt_intent):
                            intent_completion_mask[completion_idx + len(tokenized_prompt):] = torch.ones(1)
                            break
                    completion_mask[len(tokenized_prompt)-1:] = 1
                    # print(intent_completion_mask)
                    all_intent_completion_masks[gt_intent] = intent_completion_mask
                    model_inputs["all_intent_completion_masks"].append(all_intent_completion_masks)
                    model_inputs["completion_masks"].append(list(completion_mask))
                    model_inputs["gt_intents"].append(gt_intent)

            # if len(tokenized_source) > max_length:
            #     # Trim from the end of the source until it fits in the max length.
            #     to_trim = len(tokenized_source) - max_length
            #     tokenized_source = tokenized_source[:-to_trim]
            #     labels = labels[:-to_trim]
            #     logger.warning(
            #         "Truncating source on right from %d to %d tokens. Result: %s",
            #         max_length + to_trim,
            #         max_length,
            #         self.tokenizer.decode(tokenized_source),
            #     )
            #     if to_trim >= len(tokenized_completion):
            #         logger.warning(
            #             "^^^ The above truncated the entire "
            #             "completion! Skipping loading this batch element."
            #         )
            #         continue

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
            model_inputs["num_intents"].append(num_intents)
            if self.add_intent_gist_token or self.add_slot_gist_token:
                # print(completion)
                # print(ctg_slot_rev_indices)
                # print(all_ctg_completion_masks)
                # print('\n')
                model_inputs["ctg_slot_rev_indices"].append(ctg_slot_rev_indices)
            if self.add_intent_gist_token:
                model_inputs["intent_rev_indices"].append(intent_rev_indices)

        # Left-pad inputs, convert to tensor.
        for key, value in model_inputs.items():
            if key == "labels":
                pad_token_id = self.label_pad_token_id
            else:
                pad_token_id = self.tokenizer.pad_token_id
            # To left-pad inputs, reverse, then right-pad, then reverse.
            if key in ['uid', 'ctg_slot_rev_indices', 'all_ctg_completion_masks', 'num_slots',
                       'intent_rev_indices', 'all_intent_completion_masks', 'num_intents', 'gt_intents']:
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
            if self.add_intent_gist_token:
                model_inputs["attention_mask_gist"], model_inputs["prompt_attention_masks_gist_value"] = \
                    gist.make_intent_slot_value_gist_mask(
                        model_inputs["input_ids"],
                        model_inputs["prompt_input_ids"],
                        gt_intents=model_inputs['gt_intents'],
                        intent_gist_token=self.intent_gist_token_id,
                        slot_gist_token=self.slot_gist_token_id,
                        value_gist_token=self.ctg_val_gist_token_id,
                        completion_mask=model_inputs["completion_masks"],
                        ctg_slot_rev_indices=model_inputs["ctg_slot_rev_indices"],
                        all_ctg_completion_masks=model_inputs["all_ctg_completion_masks"],
                        all_num_slots=model_inputs["num_slots"],
                        intent_rev_indices=model_inputs["intent_rev_indices"],
                        all_intent_completion_masks=model_inputs["all_intent_completion_masks"],
                        all_num_intents=model_inputs["num_intents"],
                        mask_previous_intents=self.mask_previous_intents,
                        mask_previous_slots=self.mask_previous_slots,
                        add_chat_gist_tokens=self.add_chat_gist_tokens,
                        user_gist_token=self.user_gist_token_id,
                        system_gist_token=self.system_gist_token_id,
                        end_of_chat_token=self.end_of_chat_token_id,
                        tokenizer=self.tokenizer,
                    )
                model_inputs["prompt_attention_mask_gist"] = gist.make_slot_gist_mask(
                    model_inputs["prompt_input_ids"],
                    slot_gist_token=self.intent_gist_token_id,
                    all_num_slots=model_inputs["num_intents"],
                    mask_previous_slots=self.mask_previous_intents,
                )

            else:
                model_inputs["attention_mask_gist"], model_inputs["prompt_attention_masks_gist_value"] = \
                    gist.make_slot_value_gist_mask(
                        model_inputs["input_ids"],
                        model_inputs["prompt_input_ids"],
                        slot_gist_token=self.slot_gist_token_id,
                        value_gist_token=self.ctg_val_gist_token_id,
                        ctg_slot_rev_indices=model_inputs["ctg_slot_rev_indices"],
                        all_ctg_completion_masks=model_inputs["all_ctg_completion_masks"],
                        all_num_slots=model_inputs["num_slots"],
                        mask_previous_slots=self.mask_previous_slots,
                        add_chat_gist_tokens=self.add_chat_gist_tokens,
                        user_gist_token=self.user_gist_token_id,
                        system_gist_token=self.system_gist_token_id,
                        end_of_chat_token=self.end_of_chat_token_id,
                        tokenizer=self.tokenizer,
                        inbatch_reconstruct=self.inbatch_reconstruct_ratio > 0 and is_training,
                        reconstruct_token=self.gist_token_id,
                    )
                # print(instruction)
                # print(completion)
                # print(all_ctg_completion_masks["\"s4\": \""])
                # print(model_inputs["attention_mask_gist"][0, 0, -5])
                # print(model_inputs["prompt_attention_masks_gist_value"])
                # print(self.tokenizer('generate the api call. { "s0": "s0.0" }'))
                # print('\n')
                # exit()
                model_inputs["prompt_attention_mask_gist"] = gist.make_slot_gist_mask(
                    model_inputs["prompt_input_ids"],
                    slot_gist_token=self.slot_gist_token_id,
                    all_num_slots=model_inputs["num_slots"],
                    mask_previous_slots=self.mask_previous_slots,
                )

            # print(model_inputs["completion_input_ids"])
            # print(self.tokenizer.decode(model_inputs["completion_input_ids"][0]))
            # for _k, _v in model_inputs["prompt_attention_masks_gist_value"][0].items():
            #     print(_k)
            #     print(self.tokenizer.decode(_k))
            #
            # print('\n')

        elif self.add_slot_gist_token:
            model_inputs["attention_mask_gist"] = gist.make_slot_gist_mask(
                model_inputs["input_ids"],
                slot_gist_token=self.slot_gist_token_id,
                all_num_slots=model_inputs["num_slots"],
                mask_previous_slots=self.mask_previous_slots,
            )
            model_inputs["prompt_attention_mask_gist"] = gist.make_slot_gist_mask(
                model_inputs["prompt_input_ids"],
                slot_gist_token=self.slot_gist_token_id,
                all_num_slots=model_inputs["num_slots"],
                mask_previous_slots=self.mask_previous_slots,
            )
            # model_inputs["attention_mask_gist"] = gist_fn(
            #     model_inputs["input_ids"],
            #     self.slot_gist_token,
            # )
            # model_inputs["prompt_attention_mask_gist"] = gist_fn(
            #     model_inputs["prompt_input_ids"],
            #     self.slot_gist_token,
            # )

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


@dataclass
class PredictMaskDataCollatorForAlpacaCLM:
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
    predict_intent: bool = False
    intent_gist_token_id: int = 50258
    slot_gist_token_id: int = 50259
    ctg_val_gist_token_id: int = 50260
    gist_condition: str = "gist"
    num_gist_tokens: int = 10
    check_correctness: bool = False
    mask_previous_slots: bool = False
    add_chat_gist_tokens: bool = False
    user_gist_token_id: int = 50261
    system_gist_token_id: int = 50262
    end_of_chat_token_id: int = 50263
    multi_api: bool = False
    mask_previous_intents: bool = False
    inbatch_reconstruct_ratio: float = 0.
    reconstruct_token_id: int = 50258
    gist_token: str = "<GIST>"
    intent_gist_token: str = "<GIST_INTENT>"
    slot_gist_token: str = "<GIST_SLOT>"
    value_gist_token: str = "<GIST_VALUE>"
    user_gist_token: str = "<GIST_USER>"
    system_gist_token: str = "<GIST_SYSTEM>"
    end_of_chat_token: str = "<END_CHAT>"
    reconstruct_token: str = "<RECONSTRUCT>"
    detailed_completion_ratio: float = 0.

    def __post_init__(self):
        if self.max_length_human is None:
            self.max_length_human = self.max_length

    def insert_gist_tokens(self, instruction, input, completion, replacein_gist=True):
        # intent_gist_token = "<GIST_INTENT>"
        # slot_gist_token = "<GIST_SLOT>"
        # value_gist_token = "<GIST_VALUE>"
        # user_gist_token = "<GIST_USER>"
        # system_gist_token = "<GIST_SYSTEM>"
        # end_of_chat_token = "<END_CHAT>"
        slot_rev_indices = []
        # ctg_slot_value_dict = {}
        intent_rev_indices, intents, num_intents = {}, [], 0
        completion_str = completion
        if self.multi_api:
            assert self.predict_intent
            instruct_objs = json.loads(instruction)
            all_instruction_strs = []
            all_intent_strs = []

            num_slots = sum(map(len, [list(instruct_obj['slot_values'].keys()) for instruct_obj in instruct_objs]))
            cum_num_slots = 0
            num_intents = len(instruct_objs)
            if replacein_gist:
                for _iintent, instruct_obj in enumerate(instruct_objs):
                    new_intent = f"{instruct_obj['intent'][1:]} {intent_gist_token}"
                    instruction_str = f"{instruct_obj['description']}: {new_intent}: "
                    all_intent_strs.append(
                        f"{instruct_obj['description']}: {new_intent}")
                    intents.append(f'"intent": "{new_intent}",')
                    intent_rev_indices[f'"intent": "{new_intent}",'] = num_intents - _iintent
                    slot_values = []
                    slot_values_dict = instruct_obj['slot_values']

                    for _is, (slot, value) in enumerate(slot_values_dict.items()):
                        new_slot = f"{new_intent} {slot.split('.')[-1][1:]} {slot_gist_token}"
                        if 'categorical_values' in value:
                            ctg_slots.append(f'"{new_slot}": "')
                            ctg_slot_rev_indices[f'"{new_slot}": "'] = num_slots - _is - cum_num_slots
                            ctg_values = []
                            for v_idx, _val in value['categorical_values'].items():
                                v_idx = v_idx.split(".")[-1]
                                if self.add_ctg_val_gist_token:
                                    ctg_values.append(f"{_val}: {v_idx} {value_gist_token}")
                                else:
                                    ctg_values.append(f"{_val}: {v_idx}")
                            ctg_values = '(' + ', '.join(ctg_values) + ')'
                            if self.add_slot_gist_token:
                                slot_values.append(f"{value['description']} {ctg_values}: {slot.split('.')[-1][1:]} {slot_gist_token}")
                            else:
                                slot_values.append(f"{value['description']} {ctg_values}: {slot.split('.')[-1][1:]}")
                        else:
                            if self.add_slot_gist_token:
                                slot_values.append(f"{value['description']}: {slot.split('.')[-1][1:]} {slot_gist_token}")
                            else:
                                slot_values.append(f"{value['description']}: {slot.split('.')[-1][1:]}")
                    cum_num_slots += len(list(slot_values_dict.keys()))
                    instruction_str += '{' + ', '.join(slot_values) + '}'
                    # if self.add_intent_gist_token:
                    #     # instruction_str += ' ' + f"intent {instruct_obj['intent']}: {instruct_obj['description']}. " + intent_gist_token
                    #     instruction_str += ' ' + intent_gist_token
                    all_instruction_strs.append(instruction_str)
                # instruction_str = ' '.join(all_intent_strs) + ' ' + ' '.join(all_instruction_strs)
                instruction_str = ' '.join(all_instruction_strs)

                completion_obj = json.loads(completion)
                new_completion_obj = {}
                for _param_key, _param_value in completion_obj.items():
                    if _param_key == 'intent':
                        new_completion_obj['intent'] = f"{_param_value[1:]} {intent_gist_token}"
                    else:
                        intent_idx, param_idx = _param_key.split('.')
                        intent_idx, param_idx = intent_idx[1:], param_idx[1:]
                        _new_param_key = f"{intent_idx} {intent_gist_token} {param_idx} {slot_gist_token}"
                        print(f'"{_new_param_key}": "')
                        if f'"{_new_param_key}": "' in ctg_slots:
                            value_idx = _param_value.split('.')[-1]
                            _new_param_value = f"{intent_idx} {intent_gist_token} {param_idx} {slot_gist_token} {value_idx} {value_gist_token}"
                            new_completion_obj[_new_param_key] = _new_param_value
                        else:
                            new_completion_obj[_new_param_key] = _param_value
                completion_str = json.dumps(new_completion_obj)
                # print(instruction_str)
                # print(new_completion_obj)
                # print(ctg_slot_rev_indices)
                # print(ctg_slots)
                # print(num_slots)
                # print(intent_rev_indices)
                # print(intents)
                # print(completion_str)
                # exit()

            else:
                for _iintent, instruct_obj in enumerate(instruct_objs):
                    instruction_str = f"intent {instruct_obj['intent']}: {instruct_obj['description']}. "
                    all_intent_strs.append(f"intent {instruct_obj['intent']}: {instruct_obj['description']}. {intent_gist_token}")
                    intents.append(f'"intent": "{instruct_obj["intent"]}",')
                    intent_rev_indices[f'"intent": "{instruct_obj["intent"]}",'] = num_intents - _iintent
                    slot_values = []
                    slot_values_dict = instruct_obj['slot_values']

                    for _is, (slot, value) in enumerate(slot_values_dict.items()):
                        if 'categorical_values' in value:
                            ctg_slots.append(f'"{slot}": "')
                            ctg_slot_rev_indices[f'"{slot}": "'] = num_slots - _is - cum_num_slots
                            ctg_values = []
                            for v_idx, _val in value['categorical_values'].items():
                                if self.add_ctg_val_gist_token:
                                    ctg_values.append(f"{v_idx}: {_val} {value_gist_token}")
                                else:
                                    ctg_values.append(f"{v_idx}: {_val}")
                            ctg_values = '{' + ', '.join(ctg_values) + '}'
                            if self.add_slot_gist_token:
                                slot_values.append(f"{slot}: {value['description']} {ctg_values} {slot_gist_token}")
                            else:
                                slot_values.append(f"{slot}: {value['description']} {ctg_values}")
                        else:
                            if self.add_slot_gist_token:
                                slot_values.append(f"{slot}: {value['description']} {slot_gist_token}")
                            else:
                                slot_values.append(f"{slot}: {value['description']}")

                    cum_num_slots += len(list(slot_values_dict.keys()))
                    instruction_str += ', '.join(slot_values)
                    if self.add_intent_gist_token:
                        # instruction_str += ' ' + f"intent {instruct_obj['intent']}: {instruct_obj['description']}. " + intent_gist_token
                        instruction_str += ' ' + intent_gist_token
                    all_instruction_strs.append(instruction_str)
                instruction_str = ' '.join(all_intent_strs) + ' ' + ' '.join(all_instruction_strs)
            # print(instruction_str)
            # print(ctg_slots)
            # print(ctg_slot_rev_indices)
            # print(num_slots)
            # exit()
        else:
            instruct_obj = json.loads(instruction)
            instruction_str = instruct_obj['description']
            if not self.predict_intent and 'intents' in instruct_obj.keys():
                all_intents = list(instruct_obj['intents'].keys())
                assert len(all_intents) == 1
                instruction_str += instruct_obj['intents'][all_intents[0]] + '. Parameters: '
            slot_values = []
            slot_values_dict = instruct_obj['slot_values']
            slot_rev_indices = []
            num_slots = len(list(slot_values_dict.keys()))
            for _is, (slot, value) in enumerate(slot_values_dict.items()):
                slot_rev_indices.append(num_slots - _is)

                if 'categorical_values' in value:
                    # ctg_slots.append(f'"{slot}": "')
                    # ctg_slot_value_dict[f'"{slot}": "'] = []
                    # slot_rev_indices.append(num_slots - _is)
                    ctg_values = []
                    for v_idx, _val in value['categorical_values'].items():
                        if self.add_ctg_val_gist_token:
                            ctg_values.append(f"{v_idx}: {_val} {self.value_gist_token}")
                        else:
                            ctg_values.append(f"{v_idx}: {_val}")
                        # ctg_slot_value_dict[f'"{slot}": "'].append(v_idx)
                    ctg_values = '{' + ', '.join(ctg_values) + '}'
                    if self.add_slot_gist_token:
                        slot_values.append(f"{slot}: {value['description']} {ctg_values} {self.slot_gist_token}")
                    else:
                        slot_values.append(f"{slot}: {value['description']} {ctg_values}")
                else:
                    if self.add_slot_gist_token:
                        slot_values.append(f"{slot}: {value['description']} {self.slot_gist_token}")
                    else:
                        slot_values.append(f"{slot}: {value['description']}")
            instruction_str += ', '.join(slot_values)
            # print(instruction_str)
            # exit()
        if input:
            if self.add_chat_gist_tokens:
                input_sentences = input.split("USER:")
                # print(input_sentences)
                input = "\"USER:" + (self.system_gist_token + " USER:").join(input_sentences[1:])
                # print(input)
                input_sentences = input.split("SYSTEM:")
                # print(input_sentences)
                input = (self.user_gist_token + " SYSTEM:").join(input_sentences) + " " + self.end_of_chat_token

            prompt = f"Instruction: {instruction_str}\n{input}"  # noqa
        else:
            prompt = f"Instruction: {instruction_str}\nOutput:"  # noqa
        return prompt, instruction_str, completion_str, slot_rev_indices, num_slots, intent_rev_indices, num_intents

    def __call__(self, batch, return_tensors=None):
        """
        JYC: We need to make sure that no example is discarded even if it's too long. Since we need to use batch_size=1
        for 6B models and we cannot discard the only example we have.
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

            use_detailed_completion = False
            if self.detailed_completion_ratio > 0:
                rand_num = np.random.random()
                if rand_num < self.detailed_completion_ratio:
                    input = f"Input: {instance['input']} Explain the slot and values.\nOutput:"
                    use_detailed_completion = True

            add_inbatch_reconstruction = False
            if self.inbatch_reconstruct_ratio > 0:
                rand_num = np.random.random()
                if rand_num < self.inbatch_reconstruct_ratio:
                    add_inbatch_reconstruction = True
                    # if self.add_slot_gist_token or self.add_ctg_val_gist_token:
                    #     input = f"Input: {instance['input']} Reconstruct the API.\nOutput:"
                    # else:
                    #     input = f"Input: {instance['input']} Reconstruct the API.\nOutput:"

            num_slots, num_intents = 0, 0
            if self.add_intent_gist_token or self.add_slot_gist_token:
                prompt, instruction, completion, slot_rev_indices, num_slots, intent_rev_indices, num_intents \
                = self.insert_gist_tokens(
                    instruction,
                    input if input else None,
                    completion
                )
            else:
                if instance["input"]:
                    prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\n{input}" \
                             f""  # noqa
                else:
                    prompt = f"Instruction: {instance['instruction']}\n{maybe_gist_str}\nOutput:"  # noqa


            if self.detailed_completion_ratio > 0:
                # rand_num = np.random.random()
                if use_detailed_completion:
                    completion = json.dumps(json.loads(instance['output'])['detailed_output'])
                    # input = f"Input: {instance['input']} and explain the slot and values.\nOutput:"
                else:
                    completion = json.dumps(json.loads(instance['output'])['output'])

            if self.inbatch_reconstruct_ratio > 0:
                # rand_num = np.random.random()
                if add_inbatch_reconstruction:
                    if self.add_slot_gist_token or self.add_ctg_val_gist_token:
                        completion += f" {self.gist_token} {instruction.replace(f' {self.slot_gist_token}', '').replace(f' {self.value_gist_token}', '')}"
                        # input = f"Input: {instance['input']} and reconstruct the API.\nOutput:"
                    else:
                        completion += f" {self.reconstruct_token} {instruction}"
                        # input = f"Input: {instance['input']} and reconstruct the API.\nOutput:"

            # print(prompt)
            # print(completion)
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

            # if len(tokenized_instruction)

            # tokenized_source = tokenized_prompt + tokenized_completion
            # labels = [self.label_pad_token_id] * len(
            #     tokenized_prompt
            # ) + tokenized_completion

            tokenized_source = tokenized_instruction + tokenized_gist_tokens + tokenized_input + tokenized_completion
            if len(tokenized_source) > max_length:
                to_trim = len(tokenized_source) - max_length
                max_prompt_len = max_length - len(tokenized_completion)
                # ideal_max_instruction_length = int(max_source_length * 3/4)
                min_input_length = 256 # min(256, int(1/3*max_prompt_len))
                min_instruction_length = 256
                max_completion_length = max_length - min_input_length - min_instruction_length
                # print(len(tokenized_instruction))
                if len(tokenized_completion) >= max_completion_length:
                    print('trimming completion from right')
                    tokenized_completion = tokenized_completion[:-to_trim]
                elif len(tokenized_input) <= min_input_length:
                    print('trimming instruction from right')
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
                # print(max_length)
                assert len(tokenized_prompt) + len(tokenized_completion) == max_length, \
                    (len(tokenized_instruction), len(tokenized_gist_tokens), len(tokenized_input), len(tokenized_completion))

            tokenized_source = tokenized_prompt + tokenized_completion
            labels = [self.label_pad_token_id] * len(
                tokenized_prompt
            ) + tokenized_completion

            completion_mask = torch.zeros(len(tokenized_source), dtype=torch.int)
            completion_mask[len(tokenized_prompt):] = 1
            completion_mask = list(completion_mask.detach().cpu().numpy())

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
            model_inputs["num_intents"].append(num_intents)
            model_inputs["completion_mask"].append(completion_mask)

            # if self.add_intent_gist_token or self.add_slot_gist_token:
            #     model_inputs["all_slot_rev_indices"].append(slot_rev_indices)
            # if self.add_intent_gist_token:
            #     model_inputs["all_intent_rev_indices"].append(intent_rev_indices)

        # Left-pad inputs, convert to tensor.
        for key, value in model_inputs.items():
            if key == "labels":
                pad_token_id = self.label_pad_token_id
            else:
                pad_token_id = self.tokenizer.pad_token_id
            # To left-pad inputs, reverse, then right-pad, then reverse.
            if key in ['uid', 'all_slot_rev_indices', 'num_slots',
                       'all_intent_rev_indices', 'num_intents', 'gt_intents']:
                model_inputs[key] = value
                assert len(value) == 1, f"{key}: Need to left pad all_ctg_completion_masks if bsz > 1."
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

        model_inputs["prompt_slot_gist_only_mask"] = (model_inputs["prompt_input_ids"] == self.slot_gist_token_id)[:, None]
        model_inputs["slot_gist_only_mask"] = (model_inputs["input_ids"] == self.slot_gist_token_id)[:, None]
        if self.add_slot_gist_token and self.add_ctg_val_gist_token:
            if self.add_intent_gist_token:
                raise NotImplementedError
                model_inputs["attention_mask_gist"], model_inputs["prompt_attention_masks_gist_value"] = \
                    gist.make_intent_slot_value_gist_mask(
                        model_inputs["input_ids"],
                        model_inputs["prompt_input_ids"],
                        gt_intents=model_inputs['gt_intents'],
                        intent_gist_token=self.intent_gist_token_id,
                        slot_gist_token=self.slot_gist_token_id,
                        value_gist_token=self.ctg_val_gist_token_id,
                        completion_mask=model_inputs["completion_masks"],
                        ctg_slot_rev_indices=model_inputs["ctg_slot_rev_indices"],
                        all_ctg_completion_masks=model_inputs["all_ctg_completion_masks"],
                        all_num_slots=model_inputs["num_slots"],
                        intent_rev_indices=model_inputs["intent_rev_indices"],
                        all_intent_completion_masks=model_inputs["all_intent_completion_masks"],
                        all_num_intents=model_inputs["num_intents"],
                        mask_previous_intents=self.mask_previous_intents,
                        mask_previous_slots=self.mask_previous_slots,
                        add_chat_gist_tokens=self.add_chat_gist_tokens,
                        user_gist_token=self.user_gist_token_id,
                        system_gist_token=self.system_gist_token_id,
                        end_of_chat_token=self.end_of_chat_token_id,
                        tokenizer=self.tokenizer,
                    )
                model_inputs["prompt_attention_mask_gist"] = gist.make_slot_gist_mask(
                    model_inputs["prompt_input_ids"],
                    slot_gist_token=self.intent_gist_token_id,
                    all_num_slots=model_inputs["num_intents"],
                    mask_previous_slots=self.mask_previous_intents,
                )
            else:
                model_inputs["attention_mask_gist"], model_inputs["prompt_all_attention_masks_w_valuegist"], \
                model_inputs["all_attention_masks_w_valuegist"] = \
                    gist.make_slot_value_gist_mask_predmask(
                        model_inputs["input_ids"],
                        model_inputs["prompt_input_ids"],
                        slot_gist_token=self.slot_gist_token_id,
                        value_gist_token=self.ctg_val_gist_token_id,
                        all_num_slots=model_inputs["num_slots"],
                        mask_previous_slots=self.mask_previous_slots,
                        add_chat_gist_tokens=self.add_chat_gist_tokens,
                        user_gist_token=self.user_gist_token_id,
                        system_gist_token=self.system_gist_token_id,
                        end_of_chat_token=self.end_of_chat_token_id,
                        tokenizer=self.tokenizer,
                        inbatch_reconstruct=self.inbatch_reconstruct_ratio > 0,
                        reconstruct_token=self.gist_token_id,
                    )

                model_inputs["prompt_attention_mask_gist"] = gist.make_slot_gist_mask(
                    model_inputs["prompt_input_ids"],
                    slot_gist_token=self.slot_gist_token_id,
                    all_num_slots=model_inputs["num_slots"],
                    mask_previous_slots=self.mask_previous_slots,
                )
                # print(model_inputs["prompt_attention_mask_gist"][0, 0, -1])
            # print(model_inputs["completion_input_ids"])
            # print(self.tokenizer.decode(model_inputs["completion_input_ids"][0]))
            # for _k, _v in model_inputs["prompt_attention_masks_gist_value"][0].items():
            #     print(_k)
            #     print(self.tokenizer.decode(_k))
            #
            # print('\n')

        elif self.add_slot_gist_token:
            model_inputs["attention_mask_gist"] = gist.make_slot_gist_mask(
                model_inputs["input_ids"],
                slot_gist_token=self.slot_gist_token_id,
                all_num_slots=model_inputs["num_slots"],
                mask_previous_slots=self.mask_previous_slots,
            )
            model_inputs["prompt_attention_mask_gist"] = gist.make_slot_gist_mask(
                model_inputs["prompt_input_ids"],
                slot_gist_token=self.slot_gist_token_id,
                all_num_slots=model_inputs["num_slots"],
                mask_previous_slots=self.mask_previous_slots,
            )
            # model_inputs["attention_mask_gist"] = gist_fn(
            #     model_inputs["input_ids"],
            #     self.slot_gist_token,
            # )
            # model_inputs["prompt_attention_mask_gist"] = gist_fn(
            #     model_inputs["prompt_input_ids"],
            #     self.slot_gist_token,
            # )

        else:
            model_inputs["attention_mask_gist"] = gist_fn(
                model_inputs["input_ids"],
                self.gist_token_id,
                inbatch_reconstruct=self.inbatch_reconstruct_ratio > 0,
                reconstruct_token=self.reconstruct_token_id,
            )
            model_inputs["prompt_attention_mask_gist"] = gist_fn(
                model_inputs["prompt_input_ids"],
                self.gist_token_id,
            )

        return model_inputs
