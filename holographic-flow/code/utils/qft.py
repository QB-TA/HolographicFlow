import torch

def phi4_action(x, j, mu, lam, ext=0.):
    '''
    dim(phi) = [batch, ch, x, y]
    '''
    action = - j * (x.conj() * x.roll(1,dims=2) +
                    x.conj() * x.roll(1,dims=3))    
    action = action.sum(dim=(2,3))

    x_sq = x.conj() * x

    action += mu * x_sq.sum(dim=(2,3))
    action += lam * x_sq.square().sum(dim=(2,3))

    ext = torch.tensor(ext, dtype=x.dtype)
    action -= ((ext.conj()*x + x.conj()*ext) / 2).sum(dim=(2,3)) 

    action = action.sum(dim=1)

    return action.real