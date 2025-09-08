import random
import numpy as np
from basictensor_cuda import BasicTensor

class Module :
    def zero_grad(self) :
        for p in self.parameters() :
            p.grad = 0

    def parameters(self) :
        return []

class Layer(Module) :
    def __init__(self,nin,nout,nonlin=True):
        self.nin = nin
        self.nout = nout
        self.W = BasicTensor(np.random.randn(nout,nin) * np.sqrt(2.0/nin))
        self.b = BasicTensor(np.zeros(nout))
        self.nonlin = nonlin

    def __call__(self,x):
        x = x if isinstance(x, BasicTensor)  else BasicTensor(x)

        act = x @ self.W.T + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        act = 'ReLU' if self.nonlin else 'Linear'
        return f"{act} Layer({self.nin} -> {self.nout})"

class MLP(Module):
    def __init__(self, nin, nouts) :
        self.nouts = nouts
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)) : 
            nonlin = i != len(nouts) - 1
            self.layers.append(Layer(sz[i],sz[i+1],nonlin = nonlin))

    def __call__(self,x) :
        for layer in self.layers : 
            x = layer(x)
        return x

    def parameters(self) :
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) :
        for p in self.parameters() : 
            p.grad = np.zeros_like(p.value)

if __name__ == "__main__" :
    x = [BasicTensor(1.0), BasicTensor(-2.0), BasicTensor(3.0)]
    layer = Layer(3,2,nonlin=True)
    out = layer(x)

    print(layer)
    print(out)
