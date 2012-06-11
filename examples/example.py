from math import sin, pi
from numba.decorators import vectorize

#@vectorize
def sinc(x):
    rval = None
    if x == 0.0:
        rval = 1.0
    else:
        rval = sin(x * pi) / (pi * x)
    print rval
    return rval


from numba.translate import Translate
t = Translate(sinc)
t.translate(verbose=True)
print t.mod
sinc = t.make_ufunc()

#sinc = vectorize(sinc)

from numpy import linspace
x = linspace(-5,5,1001)
y = sinc(x)
from pylab import plot, show
plot(x,y)
show()
