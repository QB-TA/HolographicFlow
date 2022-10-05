def phi4_action(x, k, m, lam):
    '''
    dim(phi) = [batch, ch, x, y]
    '''
    action = - k * (x.conj() * x.roll(1,dims=2) +
                    x.conj() * x.roll(1,dims=3))    
    action = action.sum(dim=(2,3))

    x_sq = x.conj() * x

    action += m * x_sq.sum(dim=(2,3))
    action += lam * x_sq.square().sum(dim=(2,3))
    action = action.sum(dim=1)

    return action.real