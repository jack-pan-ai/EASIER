import torch
import easier as esr
from easier.core.jit import EasierTracer
from torch.nn import Module
import torch.nn as nn

class Node1(Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: esr.Tensor):
        c = self.a * x + self.b
        # d = c.clone()
        e = c + self.b
        return e

class Node2(Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: esr.Tensor):
        c = self.a * x + self.b
        # d = c.clone()
        e = c + self.b
        return e

class MyModel(esr.Module):
    def __init__(self):
        super().__init__()
        n = 6   # number of vertices
        m = 9   # number of edges

        self.vertex = esr.Tensor(torch.randn(n, 2), mode='partition')
        self.c = esr.Tensor(torch.zeros((n, 2)), mode='partition')

        self.node1 = Node1(a=1., b=2.)     
        self.node2 = Node2(a=4., b=5.)   
        self.d = 0.5
        self.array = esr.Tensor(torch.randn(n, 2), mode='partition')

    def forward(self):
        aa = self.node1(self.vertex) 
        bb = self.node2(self.vertex)   
        
        self.array[:] = self.d * aa + (1 - self.d) * bb

if __name__ == "__main__":
    torch.manual_seed(0)
    model = MyModel()

    tracer = EasierTracer()
    graph = tracer.trace(model)
    graph.print_tabular()
    