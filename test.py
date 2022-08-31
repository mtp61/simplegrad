from simplegrad import Num, display_dag, relu, log, exp


a = Num(2)
b = Num(3)
c = a - b
d = relu(c)
e = d + a
f = e * e
g = Num(5)
h = f / a
i = g + h
j = relu(i)
k = log(j)
l = exp(k)

a.name = 'a'
b.name = 'b'
c.name = 'c'
d.name = 'd'
e.name = 'e'
f.name = 'f'
g.name = 'g'
h.name = 'h'
i.name = 'i'
j.name = 'j'
k.name = 'k'
l.name = 'l'

display_dag(l, scale=.35)

