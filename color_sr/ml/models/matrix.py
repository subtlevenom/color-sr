from typing import Dict, List
from torch import nn
import graphkit as gk


class Matrix(nn.Module):

    def __init__(self, modules: Dict[str, nn.Module], metadata: List):
        super(Matrix, self).__init__()
        self.nodes = nn.ModuleDict(modules)
        self.graph = self.compose(metadata)

    def compose(self, metadata):
        ops = [
            gk.operation(
                name=f'{i}.{m.module}', needs=m.inputs,
                provides=m.outputs)(self.nodes[m.module])
            for i, m in enumerate(metadata)
        ]
        return gk.compose(name='Matrix')(*ops)

    def forward(self, **kwargs):
        return self.graph(kwargs)
