import networkx as nx
import matplotlib.pyplot as plt
import random
import math


class Num:
    unary_operations = ['relu', 'log', 'exp']
    binary_operations = ['add', 'sub', 'mul', 'div']

    def __init__(self, value: float, prev=None, name=''):
        self.value = float(value)
        self.prev = prev # tuple of (operation, num1, num2)
        if len(name) > 0:
            self.name = name
        else:
            self.name = ''
            for _ in range(3):
                self.name += random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G',
                        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                        'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        self.grad: float = 0

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
        # find topological ordering
        to_visit = [self]
        visited = set()
        order = []
        def dfs(node):
            if node.prev is not None:
                dfs(node.prev[1])
                if node.prev[0] in self.binary_operations:
                    dfs(node.prev[2])
            if node not in order:
                order.append(node)
        dfs(self)
        order.reverse()

        # update gradients
        self.grad = 1
        for node in order:
            if node.prev is not None:
                # update gradient for previous nodes
                if node.prev[0] in self.unary_operations:
                    p1 = node.prev[1]
                    if node.prev[0] == 'relu':
                        grad1 = 1 if p1.value > 0 else 0
                    elif node.prev[0] == 'log':
                        grad1 = 1 / p1.value
                    elif node.prev[0] == 'exp':
                        grad1 = math.exp(p1.value)
                    p1.grad += node.grad * grad1
                elif node.prev[0] in self.binary_operations:
                    p1 = node.prev[1]
                    p2 = node.prev[2]
                    if node.prev[0] == 'add':
                        grad1 = 1
                        grad2 = 1
                    if node.prev[0] == 'sub':
                        grad1 = 1
                        grad2 = -1
                    if node.prev[0] == 'mul':
                        grad1 = p2.value
                        grad2 = p1.value
                    if node.prev[0] == 'div':
                        grad1 = 1 / p2.value
                        grad2 = -p1.value / (p2.value ** 2)
                    p1.grad += node.grad * grad1
                    p2.grad += node.grad * grad2


def relu(num: Num) -> Num:
    value = num.value if num.value > 0 else 0
    return Num(value, prev=('relu', num))


def log(num: Num) -> Num:
    return Num(math.log(num.value), prev=('log', num))


def exp(num: Num) -> Num:
    return Num(math.exp(num.value), prev=('exp', num))


def display_dag(root: Num, scale: float=1.0, font_scale: float=1.0):
    # TODO scale graph automatically
    G = nx.DiGraph()
    G.add_node(root)
    labels_dict = {}

    def get_label(node):
        if node.prev is not None:
            if node.prev[0] in Num.unary_operations:
                operation = f'\n{node.prev[0]}({node.prev[1].value:.2f})'
            elif node.prev[0] in Num.binary_operations:
                operation = f'\n{node.prev[0]}({node.prev[1].value:.2f}, {node.prev[2].value:.2f})'
        else:
            operation = ''
        return f'{node.name}\ngrad: {node.grad:.2f}{operation}\n= {node.value:.2f}'

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
            node_size=scale * 8000, arrowsize = scale * 10, 
            font_size=scale * font_scale * 12)
    plt.show()

