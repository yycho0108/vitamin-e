from autograd import numpy as np, jacobian
#import numpy as np

def roll(x):
    pass

def unroll(x):
    pass

def E(x, y):
    data = unroll(x)
    c_tx, c_rx, l_tx, l_rx = data
    c_rx.T.dot(l_tx - c_tx)
    y = None

def sgn(x):
    Hc = None
