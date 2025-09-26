from typing import Dict, List
from torch import nn


class Meta(nn.Module):

    def __init__(self, modules: Dict[str, nn.Module], metadata: List):
        super(Meta, self).__init__()
        self.models = nn.ModuleDict(modules)
        self.metadata = metadata

    def forward(self, **kwargs):
        outputs = {}
        for metadata in self.metadata:
            module = self.models[metadata.module]
            args = metadata.input(kwargs)
            res = module(*args)
            res = metadata.output(res)
            kwargs = kwargs | res
            outputs = outputs | res
        return outputs
