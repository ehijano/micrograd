from module.value import Value
import random

class Neuron:
    def __init__(self, nin):
        # Input weights
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # intercept
        self.b = Value(random.uniform(-1, 1))


    def __call__(self, x):
        # w * x + b
        act = sum([wi*xi for wi,xi in zip(self.w, x)], self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs if len(outs) > 1 else outs[0]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    

class MLP:
    """ Multi Layer Perceptron """
    def __init__(self,nin, nouts):
        # nouts is an array
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
