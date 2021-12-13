import numpy as np

# Generic Gaussian #
def G(x, y, s):
    return np.exp(-(x ** 2 + y ** 2) / s)

# PDE FUNCTIONS #
def f(u, b1,b2, eps1, alp,eps2, s1,s2, ratio_temp,ratio_tiempo):
    return s1(u) * b1 * np.exp(u / (1 + eps1 * u)) + s2(u) * b2 *  ratio_temp/ratio_tiempo * np.exp(ratio_temp * u / (1 + eps2 * ratio_temp * u))-alp*u

def g(u, b, eps, q, s):
    return -s(u) * (eps / q) * b * np.exp(u / (1 + eps * u)) 


def H(u, upc):
    S = np.zeros_like(u)
    S[u >= upc] = 1.0
    return S

# Fuel boundary
def b0_bc(B):
    rows, cols = B.shape
    B[ 0,:] = np.zeros(cols)
    B[-1,:] = np.zeros(cols)
    B[:, 0] = np.zeros(rows)
    B[:,-1] = np.zeros(rows)
    return B