# Library imports
import numpy as np
import matplotlib.pyplot as plt
import utils
from time_solver import *
import time
import matplotlib.animation as animation
import os 
import imageio
from PIL import Image
from statistics import mean

# Function for boundary conditions
def boundaryConditions(U, B1,B2):
    Ub = np.copy(U)
    B1b = np.copy(B1)
    B2b = np.copy(B2)
    Ny, Nx = Ub.shape
    # Only Dirichlet: 
    # Temperature
    Ub[ 0,:] = np.zeros(Nx)
    Ub[-1,:] = np.zeros(Nx)
    Ub[:, 0] = np.zeros(Ny)
    Ub[:,-1] = np.zeros(Ny)
    # Fuel
    B1b[0 ,:] = np.zeros(Nx)
    B1b[-1,:] = np.zeros(Nx)
    B1b[:, 0] = np.zeros(Ny)
    B1b[:,-1] = np.zeros(Ny)
    # Fuel
    B2b[0 ,:] = np.zeros(Nx)
    B2b[-1,:] = np.zeros(Nx)
    B2b[:, 0] = np.zeros(Ny)
    B2b[:,-1] = np.zeros(Ny)
    return Ub, B1b,B2b

# Right hand side of the equations
def RHS(t, r, **kwargs):
    # Parameters 
    X, Y = kwargs['x'], kwargs['y']
    x, y = X[0], Y[:, 0] 
    V   = kwargs['V']
    kap = kwargs['kap']
    f = kwargs['f']
    g1 = kwargs['g1']
    g2 = kwargs['g2']
    Nx = x.shape[0]
    Ny = y.shape[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Vector field evaluation
    V1, V2 = V(x, y, t)
    
    # Recover u and b from vector. Reshape them into matrices
    U = np.copy(r[:Ny * Nx].reshape((Ny, Nx)))
    B1 = np.copy(r[Ny * Nx:2*Ny * Nx].reshape((Ny, Nx)))
    B2 = np.copy(r[2*Ny * Nx:].reshape((Ny, Nx)))

    # Compute derivatives #
    Ux = np.zeros_like(U)
    Uy = np.zeros_like(U)
    Uxx = np.zeros_like(U)
    Uyy = np.zeros_like(U)
    # First derivative (forward finite difference)
    Ux[1:-1, 1:-1] = (U[1:-1, 1:-1] - U[1:-1, :-2]) / dx
    Uy[1:-1, 1:-1] = (U[1:-1, 1:-1] - U[:-2, 1:-1]) / dy
    # # First derivatives (central finite difference)
    # Ux[1:-1, 1:-1] = (U[1:-1, 2:] - U[1:-1, :-2]) / 2 / dx
    # Uy[1:-1, 1:-1] = (U[2:, 1:-1] - U[:-2, 1:-1]) / 2 / dy
    # Second derivatives (central finite difference)
    Uxx[1:-1, 1:-1] = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dx / dx
    Uyy[1:-1, 1:-1] = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dy / dy

    # Laplacian of u
    lapU = Uxx + Uyy
    # Compute diffusion term
    diffusion = kap * lapU # \kappa \Delta u

    # Compute convection term
    convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
    # Compute reaction term
    reaction = f(U, B1,B2) # eval fuel
    
    # Compute RHS
    Uf = diffusion - convection + reaction # Temperature
    B1f = g1(U, B1) # Fuel
    B2f = g2(U, B2) # Fuel

    # Add boundary conditions
    Uf, B1f, B2f = boundaryConditions(Uf, B1f,B2f)

    # Build \mathbf{y} = [vec(u), vec(\beta)]^T and return
    return np.r_[Uf.flatten(), B1f.flatten(),B2f.flatten()] 

### PARAMETERS ###
# Model parameters #
kap = 1e-1 # coef. de difusión
eps1 = 3e-1 # energía de activación escalada
eps2 = 3.5e-1
alp = 1e-3 # coef. de enfriamiento de Newton(como se enfría verticalmente)
q1 = 1 # calor de reacción
q2 = 1.5
ratio_temp=eps2/eps1 # ratio entre energía de activación
ratio_tiempo=(eps2/q2*np.exp(1/eps2))/(eps1/q1*np.exp(1/eps1) ) 
upc1 = 3 # temp. donde empieza a quemarse
upc2 = 3
x_min, x_max = 0, 90
y_min, y_max = 0, 90
t_min, t_max = 0, 20

# Re-define PDE funtions with parameters #
s1 = lambda u: utils.H(u, upc1)
s2 = lambda u: utils.H(u, upc2)
ff = lambda u, b1,b2: utils.f(u, b1,b2, eps1, alp, eps2, s1,s2,ratio_temp, ratio_tiempo)
gg1 = lambda u, b: utils.g(u, b, eps1, q1, s1)
gg2 = lambda u, b: utils.g(u*ratio_temp, b, eps2/ratio_tiempo, q2, s2)

# Numerical #
# Space nodes
Nx = 128
Ny = 128
# Time nodes
Nt = 2000

# Domain #
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
t = np.linspace(t_min, t_max, Nt)
X, Y = np.meshgrid(x, y)

# Initial conditions
u0 = lambda x, y: 6 * utils.G(x - 20, y - 20, 20) # Temperature
b0 = lambda x, y: x>y # Fuel
b1 = lambda x, y: x<=y # Fuel
w1 = lambda x, y, t:  np.cos(np.pi/4 + x * 0) # Wind 
w2 = lambda x, y, t:  np.sin(np.pi/4 + x * 0) # Wind

# Wind effect
V = lambda x, y, t: (w1(x, y, t), w2(x, y, t))

# Just log
print("Nx =", Nx)
print("Ny =", Ny)
print("Nt =", Nt)
print("dx =", x[1] - x[0])
print("dy =", y[1] - y[0])
print("dt =", t[1] - t[0])

# Parameters #
# Domain, functions, etc.
params = {'x': X, 'y': Y, 'V': V, 'kap': kap, 'f': ff, 'g1': gg1,'g2': gg2,}

# Animation #
# Initial condition (vectorized)
y0 = np.r_[u0(X, Y).flatten(), b0(X, Y).flatten(), b1(X, Y).flatten()]

# Mask RHS to include parameters
F = lambda t, y: RHS(t, y, **params)

# Solve IVP #
time_start = time.time()
# R = IVP(t, y0, F, 'RK45') # Use solve_ivp from scipy.integrate 
R = EulerAnimation(t, y0, F) # Use RK4 'from scratch'
time_end = time.time()
solve_time = time_end - time_start
print("Time: ", solve_time, "[s]")




# Recover u and b from approximation. Reshape them into matrices
U = R[:, :Nx*Ny].reshape((Nt, Ny, Nx))
B1 = R[:, Nx*Ny:2*Nx*Ny].reshape((Nt, Ny, Nx))
B2 = R[:, 2*Nx*Ny:].reshape((Nt, Ny, Nx))

R=None

# Plot last time step approximation
filenames = []

for i in range(0,Nt,50):
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
    temp = axs[0].contourf(X, Y, U[i], cmap=plt.cm.jet)
    fuel1 = axs[1].contourf(X, Y, B1[i], cmap=plt.cm.Oranges)
    fuel2 = axs[2].contourf(X, Y, B2[i], cmap=plt.cm.Oranges)
    fig.colorbar(temp, ax=axs[0])
    fig.colorbar(fuel1, ax=axs[1])
    fig.colorbar(fuel2, ax=axs[2])
    axs[0].set_xlabel(r"$x$")
    axs[1].set_xlabel(r"$x$")
    axs[2].set_xlabel(r"$x$")
    axs[0].set_ylabel(r"$y$")
    plt.tight_layout()
    filename='step'+str(i)+'.png'
    plt.savefig(filename, dpi=100)
    filenames.append(filename)
    plt.close()

with imageio.get_writer('incendio_python.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
          
print("GIF listo!") 


# Remove files
for filename in set(filenames):
    os.remove(filename)

# Benchmarking # 

tiempo=[]

for i in range(0,10):
    # Initial condition (vectorized)
    y0 = np.r_[u0(X, Y).flatten(), b0(X, Y).flatten(), b1(X, Y).flatten()]
    # Mask RHS to include parameters
    F = lambda t, y: RHS(t, y, **params)

    # Solve IVP #
    time_start = time.time()
    # R = IVP(t, y0, F, 'RK45') # Use solve_ivp from scipy.integrate 
    R = Euler(t, y0, F) # Use RK4 'from scratch'
    time_end = time.time()
    solve_time = time_end - time_start
    tiempo.append(solve_time)
    print("Time: ", solve_time, "[s]")


print(tiempo)

print(mean(tiempo))


print("TERMINADO")






