# Nature's Cost Function: Simulating Physics by Minimizing the Action
# Tim Strang, Isabella Caruso, and Sam Greydanus | 2023 | MIT License

import numpy as np
import torch, time, argparse
from core_physics import *
from utils import *


############################# MINIMIZING THE ACTION #############################

def action(x, L_fn, dt):
    xdot = (x[1:] - x[:-1]) / dt
    xdot = torch.cat([xdot, xdot[-1:]], axis=0)
    T, V = L_fn(x, xdot)
    return T.sum()-V.sum(), T, V

def minimize_action(path, steps, step_size, L_fn, dt, opt='sgd', print_updates=15, e_coeff=0, verbose=True):
    t = np.linspace(0, len(path.x)-1, len(path.x)) * dt
    optimizer = torch.optim.SGD(path.parameters(), lr=step_size, momentum=0) if opt=='sgd' else \
                torch.optim.Adam(path.parameters(), lr=step_size)
    xs = [path.x.clone().data]
    info = {'S' : [], 'T' : [], 'V' : []}
    with torch.no_grad():
        S, T, V = action(path.x, L_fn, dt)
    E0 = T[0] + V[0] if len(T.shape) > 0 else (T + V).item()
    t0 = time.time()
    for i in range(steps):
        S, T, V = action(path.x, L_fn, dt)
        info['S'].append(S.item()) ; info['T'].append(T.sum().item()) ; info['V'].append(V.sum().item())

        E_loss = e_coeff * (E0 - (T + V)).pow(2).mean() if e_coeff != 0 else torch.tensor(0.0)
        loss = S + E_loss
        loss.backward() ; path.x.grad.data[[0,-1]] *= 0
        optimizer.step() ; path.zero_grad()

        if print_updates > 0 and i % (steps//print_updates) == 0:
            xs.append(path.x.clone().data)
            if verbose:
                print('step={:04d}, S={:.3e} J*s, E_loss={:.3e}, dt={:.1f}s'\
                      .format(i, S.item(), E_loss.item(), time.time()-t0))
            t0 = time.time()
    return t, path, xs, info

class PerturbedPath(torch.nn.Module):
    def __init__(self, x_true, N, sigma=0, shift=False, zero_basepath=False,
                    coords=2, is_ephemeris=False, clip_rng=1, k = 3):
        super(PerturbedPath, self).__init__()
        np.random.seed(1)
        self.x_true = x_true
        x_noise = sigma*np.random.randn(*x_true.shape).clip(-clip_rng, clip_rng)
        x_noise[:k] = x_noise[-k:] = 0
        if is_ephemeris:
            x_noise[:,0,:] = 0 # don't perturb the Sun
        x_basepath = np.copy(x_true)
        x_basepath[1:-1] = x_basepath[1:-1]*0 if zero_basepath else x_basepath[1:-1]
        self.x_pert = x_pert = (x_basepath + x_noise).reshape(-1, N*coords)
        if shift:
            x_pert_shift = np.concatenate([x_pert[5:-5,2:], x_pert[5:-5,:2]], axis=-1)
            self.x_pert[5:-5] = x_pert[5:-5] = x_pert_shift
            print(self.x_pert.shape)
        self.x = torch.nn.Parameter(torch.tensor(x_pert)) # [time, N*2]
