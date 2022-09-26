def phi4_action(phi, k, m, lam):
    '''
    dim(phi) = [batch, ch, x, y]
    '''
    action = - k * (phi.conj() * phi.roll(1,dims=2) +
                    phi.conj() * phi.roll(1,dims=3))    
    action = action.sum(dim=(2,3))

    phi_sq = phi.conj() * phi

    action += m * phi_sq.sum(dim=(2,3))
    action += lam * phi_sq.square().sum(dim=(2,3))
    action = action.sum(dim=1)

    return action.real