import numpy as np 


def lrp(layer, a, R):
    clone = layer.clone()
    clone.W = max(0, layer.W )
    clone.B = 0

    z = clone.forward(a)
    s = R/z
    c = clone.backward(s)

    return a*c
