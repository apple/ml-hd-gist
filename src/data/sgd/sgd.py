#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import List
import json

import datasets
from datasets.splits import NamedSplit

logger = datasets.logging.get_logger(__name__)


class SGDConfig(datasets.BuilderConfig):
    def __init__(
        self,
        *args,
        train_file=None,
        validation_file=None,
        test_file=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_file: str = train_file
        self.validation_file: str = validation_file
        self.test_file: str = test_file


class SGD(datasets.GeneratorBasedBuilder):
    """SGD Dataset."""

    VERSION = datasets.Version("1.0.1")
    BUILDER_CONFIG_CLASS = SGDConfig
    BUILDER_CONFIGS = [
        SGDConfig(
            name="default",
            train_file="./data/sgd/train.json",
            validation_file="./data/sgd/dev.json",
            test_file="./data/sgd/test.json",  # noqa
            description="Default config for SGD",
        ),
        SGDConfig(  ## SGD Data used for training LLaMA baseline WITHOUT reconstruction
            name="d3st_prompt+date",
            train_file="./data/sgd_d3st_prompt/train.json",
            validation_file="./data/sgd_d3st_prompt/dev.json",
            test_file="./data/sgd_d3st_prompt/test.json",  # noqa
            description="Default config for SGD",
        ),
        SGDConfig(  ## SGD-X/v5 Data used for evaluating LLaMA baseline
            name="x5_d3st_prompt+date",
            train_file="./data/sgd_x_d3st_prompt/v5/train.json",
            validation_file="./data/sgd_x_d3st_prompt/v5/dev.json",
            test_file="./data/sgd_x_d3st_prompt/v5/test.json",  # noqa
            description="Default config for SGD",
        ),
        SGDConfig(  ## SGD Data used for training HD-Gist WITHOUT reconstruction
            name="d3st_prompt+date_jsonInstruct",
            train_file="./data/sgd_d3st_prompt_jsonInstruct/train.json",
            validation_file="./data/sgd_d3st_prompt_jsonInstruct/dev.json",
            test_file="./data/sgd_d3st_prompt_jsonInstruct/test.json",  # noqa
            description="Default config for SGD",
        ),
        SGDConfig(  ## SGD-X/v5 Data used for evaluating HD-Gist
            name="x5_d3st_prompt+date_jsonInstruct",
            train_file="./data/sgd_x_d3st_prompt_jsonInstruct/v5/train.json",
            validation_file="./data/sgd_x_d3st_prompt_jsonInstruct/v5/dev.json",
            test_file="./data/sgd_x_d3st_prompt_jsonInstruct/v5/test.json",  # noqa
            description="Default config for SGD",
        ),
        SGDConfig(
            name="d3st_prompt+date_debug",
            train_file="./data/sgd_d3st_prompt/train.json",
            validation_file="./data/sgd_x_d3st_prompt/v5/dev_small.json",
            test_file="./data/sgd_x_d3st_prompt/v5/test.json",  # noqa
            description="Default config for SGD",
        ),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="SGD Data",
            features=datasets.Features(
                {
                    "uid": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    # "source": datasets.Value("string"),
                    "dialogue_id": datasets.Value("string"),
                    "turn": datasets.Value("string"),
                    "split": datasets.Value("string"),
                    "service": datasets.Value("string"),
                    "function": datasets.Value("string"),
                    "parameters": datasets.Sequence(feature=datasets.Value("string")),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        del dl_manager
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": self.config.train_file,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("validation_unseen"), #datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": self.config.validation_file,
                    "split": "validation_unseen",
                },
            ),
            datasets.SplitGenerator(
                name=NamedSplit("test_unseen"),#datasets.Split.TEST,
                gen_kwargs={
                    "path": self.config.test_file,
                    "split": "test_unseen",
                },
            ),
        ]

    def _generate_examples(
        self,
        path: str,
        split: str,
    ):
        """Yields examples."""
        logger.info(f"Generating {split} tasks from = {path}")
        with open(path, encoding="utf-8") as split_f:
            task_json = json.load(split_f)
            for idx, instance in enumerate(task_json):
                instance["split"] = split
                yield f"sgd_{split}_{idx}", instance
