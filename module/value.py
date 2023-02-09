
import math
import numpy as np

# All code based on https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

class Value:
    """Special case of Pytorch Tensor. 
    Computes backpropagation automatically in order to perform optimization on loss functions.
    All functions match Pytorch API structure"""

    def __init__(self, data, _children = (), _op = '', label = ''):
        self.data = data
        
        # Derivative of L wrt variable
        self.grad = 0.0
        # Backward function
        self._backward = lambda: None

        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        # To be able to add floats / ints
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        # To be able to add floats / ints
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __truediv__(self, other):
        # To be able to add floats / ints
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += out.grad * 1.0 / other.data
            other.grad += - out.grad * self.data / (other.data**2)

        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)

        def _backward():
            x = self.data
            self.grad += out.grad * 4.0 * math.exp(2*x) / ( (math.exp(2*x) + 1)**2 )

        out = Value(t, (self,), 'tanh')

        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        t = math.exp(x)

        def _backward():
            self.grad += out.grad * out.data

        out = Value(t, (self,), 'exp')

        out._backward = _backward
        return out
    
    def __pow__(self, other):

        assert isinstance(other, (int, float))

        def _backward():
            self.grad += out.grad * other * self.data**(other - 1)

        out = Value(self.data**other, (self,), '^')

        out._backward = _backward
        return out

    def topological_sort(self):
        topo_sort = []
        visited = set()

        def build_topo(n):
            if n not in visited:
                visited.add(n)
                for child in n._prev:
                    build_topo(child)
                topo_sort.append(n)
            return topo_sort
        
        return build_topo(self)

    def backward(self):
        topo = self.topological_sort()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()