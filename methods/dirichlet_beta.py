import numpy as np
def get_beta(data_distribution):
    # data_distribution: an ndarry list of data ratio wich sum to 1
    num = data_distribution.shape[0]
    e = data_distribution.mean()
    v = data_distribution.var()
    return (e - e * e - v) / (v * num) if num > 0 else None

def get_distribution(data_counts):
    if isinstance(data_counts, list):
        data_counts = np.array(data_counts)
    # data_counts: amount of data per client
    data_counts = data_counts / data_counts.sum()
    return get_beta(data_counts)
    