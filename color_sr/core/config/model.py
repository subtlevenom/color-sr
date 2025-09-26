import torch
from pydantic import BaseModel
from enum import Enum
from typing import Any, Dict, List, Union


class Module(BaseModel):
    model: str
    params: Dict = None

class Metadata(BaseModel):
    module: str
    inputs: List[str] = None
    outputs: List[str] = None

    def input(self, kwargs):
        return [kwargs[k] for k in self.inputs]

    def output(self, args):
        if isinstance(args, torch.Tensor):
            args = args,
        return {k: v for k, v in zip(self.outputs, args)}


class Model(BaseModel):
    metadata: List[Metadata]
    modules: Dict[str, Module]
