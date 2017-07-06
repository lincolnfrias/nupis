def normalize(x):
    x_norm = (x - x.mean()) / x.std()
    return x_norm
