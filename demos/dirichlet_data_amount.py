# Degree of Difference for Quantity Skew, different amount of clients test.
# The calling method of Label Distribution Skew estimation is similar.

import sys
import os
import numpy as np
PATH_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PATH_BASE)
sys.path.append(os.path.join(PATH_BASE, 'demos'))
PATH_DATA = os.path.join(PATH_BASE, 'data')
import tools.dataset_util as du
from methods import dirichlet_beta as dbc

for n_parties in range(5, 51, 5):
    for beta_n in range(4, 20, 1):
        beta = beta_n / 10
        aver_predict_beta = []
        for _ in range(5):
            X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = du.partition_data(
                'fmnist', os.path.join(PATH_BASE, "data"), './logs/', 'iid-diff-quantity', n_parties, beta=beta)
            client_sum_counts = []
            for client_id in traindata_cls_counts:
                cnt = 0
                for cls_id in traindata_cls_counts[client_id]:
                    cnt += traindata_cls_counts[client_id][cls_id]
                client_sum_counts.append(cnt)
            aver_predict_beta.append(dbc.get_distribution(client_sum_counts))

        print(f"n_parties: {n_parties}, simulated_beta: {beta}, estimated beta_q: {np.array(aver_predict_beta).mean()}")