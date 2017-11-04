import numpy as np

def flip_and_augment_data(X, y, p=0.2, flip_dim=-1, extend_dim=0):
    """
    Flips data with probability p
    """
    
    N = X.shape[0]
    mask = np.random.random(N) < p
    X = np.append(X, np.flip(X[mask, :, :, :], axis=-1), axis=0)
    y = np.append(y, y[mask])
    print X.shape, y.shape
    return X, y