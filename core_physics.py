# Nature's Cost Function: Simulating Physics by Minimizing the Action
# Tim Strang, Isabella Caruso, and Sam Greydanus | 2023 | MIT License

import numpy as np
import torch
from functools import partial
    

############################# GETTING ACCELERATIONS FROM POTENTIALS #############################

def accelerations(xs, xdot, potential_fn, masses=1, **kwargs):
    xs.requires_grad = True
    forces = -torch.autograd.grad(potential_fn(xs), xs)[0]
    return forces / masses


############################# CLASSIC EULER ODE SOLVER #############################

def solve_ode_euler(x0, x1, dt, accel_fn, steps=100, box_width=1, damping_coeff=1.0):
    xs = [x0, x1]
    ts = [0, dt]
    xdot = (x1 - x0) / dt
    x = xs[-1]
    for i in range(steps-2):
        a = accel_fn(x, xdot)
        xdot = xdot + a*dt
        xdot *= damping_coeff
        x = x + xdot*dt
        xs.append(x)
        ts.append(ts[-1]+dt)
    return np.asarray(ts), np.stack(xs)


############################# THE SIMULATIONS THEMSELVES #############################

def simulate_freebody(dt=0.25, steps=60):
    np.random.seed(1)
    x0, x1 = np.asarray([0.]), np.asarray([2.])
    v0 = (x1 - x0) / dt
    accel_fn = lambda x, xdot: accelerations(torch.tensor(x), None, potential_fn=V_freebody).numpy()
    return solve_ode_euler(x0, x1, dt, accel_fn, steps=steps, box_width=100)

def radial2cartesian_pend(theta, l=1):
    x1 = l * np.sin(theta)
    y1 = -l * np.cos(theta)
    return np.stack([x1, y1]).T.reshape(-1,1,2)/5 + 0.5

def simulate_pend(dt=1):
    np.random.seed(1)
    x0, x1 = np.asarray([np.pi/2]), np.asarray([np.pi/2])
    def pend_accel_fn(x, xdot, g=1, m=1, l=180):
        f = -g / l * np.sin(x)
        return f/m
    return solve_ode_euler(x0, x1, dt, pend_accel_fn)


#--------------- double pendulum ---------------#

def dblpend_accel_fn(x, xdot, m1=1, m2=1, l1=1, l2=1, g=1):
    """Return the first derivatives of x and xdot"""
    theta1, theta2 = x
    theta1dot, theta2dot = xdot
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(l1*theta1dot**2*c + l2*theta2dot**2) -
             (m1+m2)*g*np.sin(theta1)) / l1 / (m1 + m2*s**2)
    z2dot = ((m1+m2)*(l1*theta1dot**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*l2*theta2dot**2*s*c) / l2 / (m1 + m2*s**2)
    forces = np.asarray([z1dot, z2dot])
    return forces / 1 # (F/m=a)

def radial2cartesian_dblpend(thetas, l1=1, l2=1): # Convert from radial to Cartesian coordinates.
    t1, t2 = thetas.T
    x1 = l1 * np.sin(t1)
    y1 = -l1 * np.cos(t1)
    x2 = x1 + l2 * np.sin(t2)
    y2 = y1 - l2 * np.cos(t2)
    return np.stack([x1, y1, x2, y2]).T.reshape(-1,2,2)/5 + 0.6

def simulate_dblpend(dt=1, steps=200):
    np.random.seed(1)
    x0 = np.asarray([3*np.pi/7, 3*np.pi/4]) ; x1 = np.copy(x0)
    return solve_ode_euler(x0, x1, dt, dblpend_accel_fn, steps=steps)


#--------------- 3 body and gas simulations ---------------#

def simulate_3body(dt=0.5, steps=100, stable_config=True):
    np.random.seed(1)
    if stable_config:
        x0 = np.asarray([[0.4, 0.3], [0.4, 0.7], [0.7, 0.5]])
        v0 = np.asarray([[0.017, -0.006], [-.012, -.012], [0.0, 0.017]])
    else:
        x0 = np.asarray([[0.4, 0.3], [0.4, 0.7], [0.7, 0.5]])
        v0 = np.asarray([[0.017, -0.006], [-.012, -.012], [0.002, 0.017]])
    x1 = x0 + dt*v0
    accel_fn = lambda x, xdot: accelerations(torch.tensor(x), None, potential_fn=V_3body).numpy()
    return solve_ode_euler(x0, x1, dt, accel_fn, steps=steps)

def simulate_gas(dt=1, N=50):
    np.random.seed(1)
    x0 = np.random.rand(N,2)*.8 + 0.1
    v0 = np.random.randn(N,2)*.00
    x1 = x0 + dt*v0
    accel_fn = lambda x, xdot: accelerations(torch.tensor(x), None, potential_fn=V_gas).numpy()
    return solve_ode_euler(x0, x1, dt, accel_fn)

def simulate_LJ(dt=0.2, N=50, damping_coeff=0.9):
    np.random.seed(1)
    x0 = np.random.rand(N,2)*.2 + 0.4
    v0 = np.random.randn(N,2)*.0
    x1 = x0 + dt*v0
    accel_fn = lambda x, xdot: accelerations(torch.tensor(x), None, potential_fn=V_LJ).numpy()
    return solve_ode_euler(x0, x1, dt, accel_fn, damping_coeff=damping_coeff)


#--------------- solar system ephemeris ---------------#

def get_coords(df, planets, i=0):
    return np.asarray([ [df[p + '_x'].iloc[i], df[p + '_y'].iloc[i]] for p in planets])

def get_masses(planets):
    d = {'sun':1.99e30, 'venus':4.87e24, 'mercury':3.3e23, 'earth':5.97e24, 'mars':6.42e23}
    return np.asarray([d[p] for p in planets])[:,None]

def simulate_planets(df, planets, dt=24*60*60, steps=365-300):
    x0 = get_coords(df, planets, i=148)
    x1 = get_coords(df, planets, i=149)
    masses = get_masses(planets)
    V_planets_fn = partial(V_planets, masses=masses)
    accel_fn = lambda x, xdot: accelerations(torch.tensor(x), None, V_planets_fn, masses).numpy()
    return solve_ode_euler(x0, x1, dt, accel_fn, steps=365-300)


############################# LAGRANGIANS #############################

def lagrangian_freebody(x, xdot, m=1):
    norm_factor = x.shape[0]
    T = (.5*m*xdot**2) / norm_factor
    V = V_freebody(x.reshape(-1, 1)) / norm_factor
    return T, V

def lagrangian_pend(x, xdot, m=1):
    norm_factor = x.shape[0]
    return T_pend(xdot) / norm_factor, V_pend(x) / norm_factor

def lagrangian_dblpend(x, xdot, m=1):
    norm_factor = x.shape[0]
    T = T_dblpend(x, xdot) / norm_factor
    V = V_dblpend(x) / norm_factor
    return T, V

def lagrangian_3body(x, xdot, m=1):
    x = x.reshape(x.shape[0], -1)
    N = x.shape[-1] // 2
    norm_factor = x.shape[0]*N
    T = (.5*m*xdot**2).sum() / norm_factor
    V = V_3body(x.reshape(-1, N, 2)).sum() / norm_factor
    return T, V

def lagrangian_gas(x, xdot, m=1):
    N = x.reshape(x.shape[0],-1).shape[1] // 2
    norm_factor = x.shape[0]*N
    T = (.5*m*xdot**2).sum() / norm_factor
    V = V_gas(x.reshape(-1, N, 2)).sum() / norm_factor
    return T, V

def lagrangian_planets(x, xdot, masses):
    N = x.reshape(x.shape[0],-1).shape[1] // 2
    norm_factor = x.shape[0]*N
    xdot = xdot.reshape(-1,N,2)
    m = torch.tensor(masses[None,:,:]) # should be ofshape [1,N,1]
    T = (.5*m*xdot**2).sum() / norm_factor
    V = V_planets(x.reshape(-1, N, 2), masses).sum() / norm_factor
    return T, V

def lagrangian_LJ(x, xdot, m=1):
    N = x.reshape(x.shape[0],-1).shape[1] // 2
    norm_factor = x.shape[0]*N
    T = (.5*m*xdot**2).sum() / norm_factor
    V = V_LJ(x.reshape(-1, N, 2)).sum() / norm_factor
    return T, V


############################# POTENTIAL FUNCTIONS #############################

def V_freebody(xs): # assume xs measured on vertical axis
    return xs

def V_pend(x, m=1, l=180, g=1):
    return -m*l*g*(torch.cos(x) - 1)

def T_pend(xdot, m=1, l=180, g=1):
    return m*l*g*(l*xdot**2) / (2 * g)

def V_dblpend(x, m1=1, m2=1, l1=1, l2=1, g=1):
    th1, th2 = x[...,0], x[...,1]
    return -(m1 + m2) * l1 * g * torch.cos(th1) - m2 * l2 * g * torch.cos(th2)
    
def T_dblpend(x, xdot, m1=1, m2=1, l1=1, l2=1, g=1):
    th1, th2 = x[...,0], x[...,1]
    th1d, th2d = xdot[...,0], xdot[...,1]
    return 0.5 * m1 * (l1 * th1d) ** 2 + 0.5 * m2 * ((l1 * th1d) ** 2 + (l2 * th2d) ** 2 +
                                                     2 * l1 * l2 * th1d * th2d * torch.cos(th1 - th2))

def V_3body(xs, eps=1e-6, overlap_radius=0.05, scale_coeff=1.3e-4):
    if len(xs.shape) > 2:
        return sum([V_3body(_xs, eps, overlap_radius, scale_coeff) for _xs in xs]) # broadcast
    else:
        dist_matrix = ((xs[:,0:1] - xs[:,0:1].T).pow(2) + (xs[:,1:2] - xs[:,1:2].T).pow(2) + eps).sqrt()
        dists = dist_matrix[torch.triu_indices(xs.shape[0], xs.shape[0], 1).split(1)]
        potentials =  (dists > overlap_radius) * 1/(dists + eps)  # 1/r^2
        potentials += (dists < overlap_radius) * (5e2*(overlap_radius - dists) + 1/overlap_radius)
        return -potentials.sum() * scale_coeff

def V_gas(xs, eps=1e-6, overlap_radius=0.05, scale_coeff=1e-5): # 1e-6 -> 500 particles
    if len(xs.shape) > 2:
        return sum([V_gas(_xs, eps, overlap_radius, scale_coeff) for _xs in xs]) # broadcast
    else:
        dist_matrix = ((xs[:,0:1] - xs[:,0:1].T).pow(2) + (xs[:,1:2] - xs[:,1:2].T).pow(2) + eps).sqrt()
        dists = dist_matrix[torch.triu_indices(xs.shape[0], xs.shape[0], 1).split(1)]
        potentials  = (dists > 1-overlap_radius) * (5e2*(overlap_radius - (1-dists)) + 1/overlap_radius) # cap
        potentials = (dists > 0.5) * (dists < 1-overlap_radius) * 1/(1-dists + eps)  # 1/(1-r) (wraparound)
        potentials += (dists > overlap_radius)* (dists < 0.5) * 1/(dists + eps)  # 1/r
        potentials += (dists < overlap_radius) * (5e2*(overlap_radius - dists) + 1/overlap_radius)  # cap
        return potentials.sum() * scale_coeff

def V_LJ(xs, eps=1e-6, overlap_radius=0.04, scale_coeff=1e-5, sigma=4e-2): # 1e-6 -> 500 particles
    if len(xs.shape) > 2:
        return sum([V_LJ(_xs, eps, overlap_radius, scale_coeff, sigma) for _xs in xs]) # broadcast
    else:
        dist_matrix = ((xs[:,0:1] - xs[:,0:1].T).pow(2) + (xs[:,1:2] - xs[:,1:2].T).pow(2) + eps).sqrt()
        dists = dist_matrix[torch.triu_indices(xs.shape[0], xs.shape[0], 1).split(1)]
        
        potentials = 10*( (sigma / dists).pow(12) - (sigma / dists).pow(6) )
        potentials = potentials.clamp(None, 10)
        potentials += (dists < overlap_radius) * (1e2*(overlap_radius - dists))
        return potentials.sum() * scale_coeff

def V_planets(xs, masses, eps=1e-10, G=6.67e-11): # # 2e-25
    if len(xs.shape) > 2:
        return sum([V_planets(_xs, masses, eps=eps, G=G) for _xs in xs]) # broadcast
    else:
        ixs = torch.triu_indices(xs.shape[0], xs.shape[0], 1).split(1)
        dist_matrix = ((xs[:,0:1] - xs[:,0:1].T).pow(2) + (xs[:,1:2] - xs[:,1:2].T).pow(2) + eps).sqrt()
        mM_matrix = torch.tensor( masses.T * masses )
        U_vals = G * mM_matrix[ixs] / dist_matrix[ixs]
        return -U_vals.sum()