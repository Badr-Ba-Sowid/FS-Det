from __future__ import annotations

import pathlib
import yaml

from dataclasses import dataclass

from typing import Any
from typing_extensions import Self

__all__ = ['TrainingConfig']

@dataclass
class TrainingParams:
    seed: int
    training_split: float
    batch_size: int
    epoch: int
    data_loader_num_workers: int
    learning_rate: float
    scheduler_steps: int
    scheduler_gamma: float

@dataclass
class Uris:
    dataset: str
    label: str
    ckpts: str

@dataclass
class FewShotParams:
    n_ways: int
    k_shots: int
    n_tasks: int

@dataclass
class DatasetParams:
    name: str
    num_classes: int

class TrainingConfig:

    @classmethod
    def from_file(cls, uri: str) -> Self:
        config_path = pathlib.Path(uri)
        with config_path.open('r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        
        return cls(data)
    
    def __init__(self, data: dict[str, Any]) -> None:
        self.uris = self._parse_paths(data.get('PATHS', {}))
        self.trainig_params = self._parse_training_params(data.get('TRAINING', {}))
        self.few_shot_params = self._parse_few_shot_params(data.get('FEWSHOT', {}))
        self.dataset_params = self._parse_dataset_params(data.get('DATASET', {}))

    def _parse_training_params(self, config: dict[str, Any]) -> TrainingParams:
        return TrainingParams(seed=int(config.get('seed', 42)),
                            training_split=float(config.get('training_split', 0.7)),
                            batch_size=int(config.get('batch_size', 10)),
                            epoch=int(config.get('epoch', 50)),
                            data_loader_num_workers=int(config.get('data_loader_num_workers', 1)),
                            learning_rate=float(config.get('learning_rate', 0.001)),
                            scheduler_steps=int(config.get('scheduler_step', 10)),
                            scheduler_gamma=float(config.get('scheduler_gamma', 0.1))
                )

    def _parse_paths(self, config: dict[str, Any]) -> Uris:
        return Uris(dataset=(config.get('dataset_source', '')), 
                    label=(config.get('label_source', '')),
                    ckpts=(config.get('check_point', ''))
                )

    def _parse_few_shot_params(self, config: dict[str, Any]) -> FewShotParams:
        return FewShotParams(n_ways=int(config.get('n_way', 0)),
                                k_shots=int(config.get('k_shots', 0)),
                                n_tasks=int(config.get('n_tasks', 0))
                            )
    
    def _parse_dataset_params(self, config: dict[str, Any]) -> DatasetParams:
        return DatasetParams(name=str(config.get('name', '')), num_classes=int(config.get('num_classes', 0)))