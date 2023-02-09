
from graphviz import Digraph


def trace(root):
    """Builds a set of all nodes and edgeds in a graph"""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges

def draw_dot(root):
    """Creates a picture of a graph involving Value classes"""
    dot = Digraph(format = 'svg', graph_attr={'rankdir':'LR'})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        dot.node(name = uid, label = '{} | value = {:.2f} | grad = {:.4f}'.format(n.label, n.data, n.grad), shape = 'record')

        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid+n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot