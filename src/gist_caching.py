#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

from .data.gist import get_gist_index, get_slot_value_gist_index


@dataclass
class GistActivations:
    last_hidden_state: torch.FloatTensor
    past_key_values: Tuple[Tuple[torch.FloatTensor]]
    # In case needed, e.g. for absolute position embeddings.
    gist_indices: Optional[torch.LongTensor] = None

    def get_single(self, batch_idx):
        # Return batch-size-1 version of each key.
        return GistActivations(
            last_hidden_state=self.last_hidden_state[batch_idx : batch_idx + 1],
            past_key_values=tuple(
                (k[batch_idx : batch_idx + 1], v[batch_idx : batch_idx + 1])
                for k, v in self.past_key_values
            ),
            gist_indices=self.gist_indices[batch_idx : batch_idx + 1]
            if self.gist_indices is not None
            else None,
        )

    @classmethod
    def from_model_outputs(
        cls,
        model_outputs: ModelOutput,
        input_ids: torch.LongTensor,
        gist_token: int,
        num_gist_tokens: int,
        cache_all: bool = False,
    ) -> GistActivations:
        """
        Computes gist activations for the model.

        If cache_all is True, actually cache everything, not just the gist
        tokens. Useful for positive control decoder models, for example.

        Returns:
            GistActivations object containing the kv activations for the gist
            tokens of the encoder, and the start position of the first gist
            token (needed to correctly compute position embeddings).
        """
        assert hasattr(model_outputs, "last_hidden_state")
        assert hasattr(model_outputs, "past_key_values")

        past_key_values = model_outputs.past_key_values
        batch_size, num_heads, seq_length, hidden_size = past_key_values[0][0].shape
        device = past_key_values[0][0].device

        if cache_all:
            assert batch_size == 1, "Can only cache all if batch size is 1 for now."
            _, gist_end = get_gist_index(
                input_ids[0], gist_token, raise_if_no_tokens=True
            )
            num_gist_tokens = gist_end - 1
            '''
            Yichen Jiang: Here, we minus one because when caching all instruction tokens 
            (with the trick of setting num_gist_tokens = 1), 
            we actually want to benchmark the model with no gist token. 
            So we do not want to cache the final gist token.
            '''

        def kv():
            return torch.zeros(
                (batch_size, num_heads, num_gist_tokens, hidden_size),
                dtype=past_key_values[0][0].dtype,
                device=device,
            )

        # Save key/values and hidden states corresponding to gist tokens only.
        last_hidden_state = torch.zeros(
            (batch_size, num_gist_tokens, model_outputs.last_hidden_state.shape[-1]),
            dtype=model_outputs.last_hidden_state.dtype,
            device=device,
        )
        # Same structure as past_key_values, but only has `num_gist_tokens` spots.
        gist_past_key_values = [(kv(), kv()) for _ in past_key_values]
        gist_starts = []
        for batch_i, input_row in enumerate(input_ids):
            # Find start and end gist position.
            if cache_all:
                gist_start, gist_end = 0, num_gist_tokens
            else:
                gist_start, gist_end = get_gist_index(
                    input_row, gist_token, raise_if_no_tokens=True
                )

            # Compute # of observed gist tokens.
            obs_num_gist_tokens = gist_end - gist_start
            if obs_num_gist_tokens != num_gist_tokens:
                raise ValueError(
                    f"Expected {num_gist_tokens} gist tokens, got "
                    f"{obs_num_gist_tokens} "
                    f"(input ids: {input_row}, start {gist_start} end {gist_end})"
                )

            # For each hidden layer, save the activations corresponding to the
            # gist token.
            for layer_i, (past_k, past_v) in enumerate(past_key_values):
                gist_past_key_values[layer_i][0][batch_i] = past_k[
                    batch_i, :, gist_start:gist_end
                ]
                gist_past_key_values[layer_i][1][batch_i] = past_v[
                    batch_i, :, gist_start:gist_end
                ]

            # Keep track of gist starts for the batch.
            gist_starts.append(gist_start)

            # Save last hidden state corresponding to gist token.
            last_hidden_state[batch_i] = model_outputs.last_hidden_state[
                batch_i, gist_start:gist_end
            ]

        return GistActivations(
            last_hidden_state=last_hidden_state,
            past_key_values=tuple(gist_past_key_values),
            gist_indices=torch.tensor(gist_starts, dtype=torch.int64, device=device),
        )

@dataclass
class SlotValueGistActivations:
    last_hidden_state: torch.FloatTensor
    past_key_values: Tuple[Tuple[torch.FloatTensor]]
    # In case needed, e.g. for absolute position embeddings.
    gist_indices: Optional[torch.LongTensor] = None

    def get_single(self, batch_idx):
        # Return batch-size-1 version of each key.
        return GistActivations(
            last_hidden_state=self.last_hidden_state[batch_idx : batch_idx + 1],
            past_key_values=tuple(
                (k[batch_idx : batch_idx + 1], v[batch_idx : batch_idx + 1])
                for k, v in self.past_key_values
            ),
            gist_indices=self.gist_indices[batch_idx : batch_idx + 1]
            if self.gist_indices is not None
            else None,
        )

    @classmethod
    def from_model_outputs(
        cls,
        model_outputs: ModelOutput,
        input_ids: torch.LongTensor,
        slot_gist_token: int,
        value_gist_token: int,
        cache_all: bool = False,
    ) -> SlotValueGistActivations:
        """
        Computes gist activations for the model.

        If cache_all is True, actually cache everything, not just the gist
        tokens. Useful for positive control decoder models, for example.

        Returns:
            GistActivations object containing the kv activations for the gist
            tokens of the encoder, and the start position of the first gist
            token (needed to correctly compute position embeddings).
        """
        assert hasattr(model_outputs, "last_hidden_state")
        assert hasattr(model_outputs, "past_key_values")

        past_key_values = model_outputs.past_key_values
        batch_size, num_heads, seq_length, hidden_size = past_key_values[0][0].shape
        device = past_key_values[0][0].device

        if cache_all:
            assert batch_size == 1, "Can only cache all if batch size is 1 for now."
            _, gist_end = get_slot_value_gist_index(
                input_ids[0],
                slot_gist_token=slot_gist_token,
                value_gist_token=value_gist_token,
                raise_if_no_tokens=True
            )
            num_gist_tokens = gist_end

        gist_starts = []
        all_num_gist_tokens, all_gist_indices = [], []
        for batch_i, input_row in enumerate(input_ids):
            # Find start and end gist position.
            if cache_all:
                raise NotImplementedError
                gist_start, gist_end = 0, num_gist_tokens
            else:
                gist_indices = get_slot_value_gist_index(
                    input_row, slot_gist_token, value_gist_token, raise_if_no_tokens=True
                )

            # Compute # of observed gist tokens.
            obs_num_gist_tokens = len(gist_indices) #gist_end - gist_start
            all_num_gist_tokens.append(obs_num_gist_tokens)
            all_gist_indices.append(gist_indices)
            # if obs_num_gist_tokens != num_gist_tokens:
            #     raise ValueError(
            #         f"Expected {num_gist_tokens} gist tokens, got "
            #         f"{obs_num_gist_tokens} "
            #         f"(input ids: {input_row}, start {gist_start} end {gist_end})"
            #     )
        max_num_gist_tokens = max(all_num_gist_tokens)
        def kv():
            return torch.zeros(
                (batch_size, num_heads, max_num_gist_tokens, hidden_size),
                dtype=past_key_values[0][0].dtype,
                device=device,
            )

        # Save key/values and hidden states corresponding to gist tokens only.
        last_hidden_state = torch.zeros(
            (batch_size, max_num_gist_tokens, model_outputs.last_hidden_state.shape[-1]),
            dtype=model_outputs.last_hidden_state.dtype,
            device=device,
        )
        # Same structure as past_key_values, but only has `num_gist_tokens` spots.
        gist_past_key_values = [(kv(), kv()) for _ in past_key_values]

        for batch_i, input_row in enumerate(input_ids):
            gist_indices = all_gist_indices[batch_i]
            # print(gist_indices)
            num_gist_tokens = all_num_gist_tokens[batch_i]
            # For each hidden layer, save the activations corresponding to the
            # gist token.
            # print(num_gist_tokens)
            for layer_i, (past_k, past_v) in enumerate(past_key_values):
                # print(layer_i)
                # print(gist_past_key_values[layer_i][0][batch_i].size())
                # print(gist_past_key_values[layer_i][1][batch_i].size())
                # print(past_k[batch_i, :, gist_indices].size())
                gist_past_key_values[layer_i][0][batch_i][:, :num_gist_tokens] = past_k[
                    batch_i, :, gist_indices
                ]
                gist_past_key_values[layer_i][1][batch_i][:, :num_gist_tokens] = past_v[
                    batch_i, :, gist_indices
                ]

            # Keep track of gist starts for the batch.
            gist_starts.append(gist_indices)

            # Save last hidden state corresponding to gist token.
            last_hidden_state[batch_i] = model_outputs.last_hidden_state[
                batch_i, gist_indices
            ]

        return SlotValueGistActivations(
            last_hidden_state=last_hidden_state,
            past_key_values=tuple(gist_past_key_values),
            gist_indices=torch.stack(gist_starts, dim=0),
            #gist_indices=torch.tensor(gist_starts, dtype=torch.int64, device=device),
        )
