from __future__ import annotations

import pathlib
import yaml

from dataclasses import dataclass

from typing import Any, Optional
from typing_extensions import Self

__all__ = ['Config', 'DatasetParams']

@dataclass
class TrainingParams:
    seed: int
    training_split: float
    epochs: int
    learning_rate: float
    scheduler_steps: int
    scheduler_gamma: float
    ckpts: str
    pretrained_uri: Optional[str]
    attention: bool

@dataclass
class TestingParams:
    dataset_path:str
    model_state: str
    k_shots: list[int]

@dataclass
class FewShotParams:
    n_ways: int
    k_shots: int
    n_tasks: int
    test_k_shots: list[int]

@dataclass
class DatasetParams:
    name: str
    num_classes: int
    dataset: str
    label: str
    text_label_source: str
    data_loader_num_workers: int
    batch_size: int
    experiment_result_uri: str


class Config:

    @classmethod
    def from_file(cls, uri: str) -> Self:
        config_path = pathlib.Path(uri)
        with config_path.open('r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        return cls(data)

    def __init__(self, data: dict[str, Any]) -> None:
        self.trainig_params = self._parse_training_params(data.get('TRAINING', {}))
        self.few_shot_params = self._parse_few_shot_params(data.get('FEWSHOT', {}))
        self.dataset_params = self._parse_dataset_params(data.get('DATASET', {}))
        self.testing_params = self._parse_testing_params(data.get('TESTING', {}))

    def _parse_training_params(self, config: dict[str, Any]) -> TrainingParams:
        return TrainingParams(seed=int(config.get('seed', 42)),
                            training_split=float(config.get('train_ratio', 0.7)),
                            epochs=int(config.get('epochs', 50)),
                            learning_rate=float(config.get('learning_rate', 0.001)),
                            scheduler_steps=int(config.get('scheduler_step', 10)),
                            scheduler_gamma=float(config.get('scheduler_gamma', 0.1)),
                            ckpts=(config.get('check_point_uri', '')),
                            pretrained_uri=(config.get('pretrained_uri', None)),
                            attention=(config.get('attention', False))
                )

    def _parse_few_shot_params(self, config: dict[str, Any]) -> FewShotParams:
        return FewShotParams(n_ways=int(config.get('n_ways', 0)),
                                k_shots=int(config.get('k_shots', 0)),
                                n_tasks=int(config.get('n_tasks', 0)),
                                test_k_shots=list(config.get('test_k_shots', []))
                            )

    def _parse_dataset_params(self, config: dict[str, Any]) -> DatasetParams:
        return DatasetParams(name=str(config.get('name', '')),
                                text_label_source=str(config.get('text_label_source', '')),
                                num_classes=int(config.get('num_classes', 0)),
                                dataset=(config.get('dataset_source', '')), 
                                label=(config.get('label_source', '')),
                                data_loader_num_workers=int(config.get('data_loader_num_workers', 1)),
                                batch_size=int(config.get('batch_size', 10)),
                                experiment_result_uri=str(config.get('experiment_result_uri', ''))
                )

    def _parse_testing_params(self, config: dict[str, Any]) -> TestingParams:
        return TestingParams(dataset_path = str(config.get('dataset_path')),
                                model_state=str(config.get('model_state', '')),
                                k_shots=self.few_shot_params.test_k_shots)
