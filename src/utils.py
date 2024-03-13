#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#


import itertools
from typing import Any, List, Union, Tuple, Optional
import numpy as np
from transformers.trainer_utils import (
    EvalPrediction,
)

def first_mismatch(a: List[Any], b: List[Any], window: int = 10):
    """Returns first mismatch as well as sublists for debugging."""
    for i, (x, y) in enumerate(itertools.zip_longest(a, b)):
        if x != y:
            window_slice = slice(i - window, i + window)
            return (x, y), (a[window_slice], b[window_slice])
    return None

class MyEvalPrediction(EvalPrediction):
    def __init__(
        self,
        uids: Union[np.ndarray, Tuple[np.ndarray]],
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        super().__init__(predictions, label_ids, inputs)
        self.uids = uids

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs, self.uids))
        else:
            return iter((self.predictions, self.label_ids, self.uids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 3:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs
        elif idx == 3:
            return self.uids
