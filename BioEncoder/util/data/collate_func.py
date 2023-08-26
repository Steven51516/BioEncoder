import numpy as np
import torch

def dgl_collate_func(x):
    d, p, y = zip(*x)
    import dgl
    d = dgl.batch(d)
    p = np.array(p)
    y = np.array(y)
    return d, torch.tensor(p), torch.tensor(y)