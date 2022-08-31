import networkx as nx
import matplotlib.pyplot as plt
import random
import math


class Num:
    unary_operations = ['relu', 'log', 'exp']
    binary_operations = ['add', 'sub', 'mul', 'div']

    def __init__(self, value: float, prev=None, name=''):
        self.value = float(value)
        self.grad: float = 0
        self.prev = prev # tuple of (operation, num1, num2)
        if len(name) > 0:
            self.name = name
        else:
            self.name = ''
            for _ in range(3):
                self.name += random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G',
                        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

    def __str__(self):
        grad = None if self.grad is None else f'{self.grad:.3f}'
        text = f'(value: {self.value:.3f}, grad: {grad})'
        return f'{self.name}: {text}' if len(self.name) > 0 else text

    def __add__(self, num):
        prev = ('add', self, num)
        return Num(self.value + num.value, prev)

    def __sub__(self, num):
        prev = ('sub', self, num)
        return Num(self.value - num.value, prev)

    def __mul__(self, num):
        prev = ('mul', self, num)
        return Num(self.value * num.value, prev)

    def __truediv__(self, num):
        prev = ('div', self, num)
        return Num(self.value / num.value, prev)

    def zero_grad(self):
        self.grad = 0

    def backward(self):
        # TODO
        return


def relu(num: Num) -> Num:
    value = num.value if num.value > 0 else 0
    return Num(value, prev=('relu', num))


def log(num: Num) -> Num:
    return Num(math.log(num.value), prev=('log', num))


def exp(num: Num) -> Num:
    return Num(math.exp(num.value), prev=('exp', num))


def display_dag(root: Num, scale: float=1.0):
    G = nx.DiGraph()
    G.add_node(root)
    labels_dict = {}

    def get_label(node):
        if node.prev is not None: # TODO support relu
            if node.prev[0] in Num.unary_operations:
                operation = f'\n{node.prev[0]}({node.prev[1].value:.1f})'
            elif node.prev[0] in Num.binary_operations:
                operation = f'\n{node.prev[0]}({node.prev[1].value:.1f}, {node.prev[2].value:.1f})'
        else:
            operation = ''
        return f'{node.name}{operation}\n= {node.value:.1f}'

    # traverse
    to_visit = set([root])
    visited = set()
    while len(to_visit) > 0:
        node = to_visit.pop()
        if node.prev is not None:
            p1  = node.prev[1]
            G.add_node(p1)
            G.add_edge(p1, node)
            if p1 not in visited:
                to_visit.add(p1)
            if node.prev[0] in Num.binary_operations:
                p2 = node.prev[2]
                G.add_node(p2)
                G.add_edge(p2, node)
                if p2 not in visited:
                    to_visit.add(p2)

        labels_dict[node] = get_label(node)
        visited.add(node)

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, labels=labels_dict, with_labels=True,
            node_size=scale * 6000, arrowsize = scale * 10, 
            font_size=scale * 12)
    plt.show()

