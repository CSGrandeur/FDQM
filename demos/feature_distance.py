# Comparing feature GMM between "homo" and "real" with femnist dataset.

import sys
import os
import numpy as np
from skimage import io
PATH_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PATH_BASE)
sys.path.append(os.path.join(PATH_BASE, 'demos'))
PATH_DATA = os.path.join(PATH_BASE, 'data')
import tools.dataset_util as du
import tools.utils_local as ul
from methods.feature_distribution import FeatureSummary

def make_features(data_source='femnist', partition_type='real'):
    file_prefix = os.path.join(PATH_DATA, f"{data_source}_{partition_type}")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = du.partition_data(
        data_source, PATH_DATA, './logs/', partition_type, 10, beta=0.5)
    X_train = du.pad_data(X_train, channel='nchw')

    fs = FeatureSummary(use_gpu=False)
    gs_res = {} 
    with open(file_prefix + "_res.csv", "w") as f:
        header = ",".join(["label\\client"] + [f"{i}_aver,{i}_var" for i in traindata_cls_counts]) + "\n"
        f.write(header)
        for label in np.unique(y_train):
            line = str(label)
            label_idx_list = np.where(y_train==label)[0]  
            gs_res[label] = {}
            for client in net_dataidx_map:
                client_idx_list = net_dataidx_map[client] 
                client_label_id_list = np.intersect1d(label_idx_list, client_idx_list) 
                data_selected = X_train[client_label_id_list]
                print(label, client, data_selected.shape, len(client_label_id_list))
                ft = fs.get_feature(data_selected)
                gs_res[label][client] = {
                    'mean': ft.mean(axis=0).cpu().detach().numpy(),
                    'var': ft.var(axis=0).cpu().detach().numpy(),
                    'image': data_selected.mean(axis=0)
                }
                img = gs_res[label][client]['image']
                img = img.transpose(1, 2, 0)
                mean_v = ft.mean().cpu().detach().numpy()
                var_v = ft.var().cpu().detach().numpy()
                line += ",%.8f,%.8f" % (mean_v, var_v)
            f.write(line + "\n")
    ul.save_pickle(gs_res, file_prefix + "_gs_res.pkl")

def distance_gs_res(data_source, partition_type):
    gs_res = ul.load_pickle(os.path.join(PATH_DATA, f"{data_source}_{partition_type}_gs_res.pkl"))
    i_label = 0
    print(f"#####{data_source}_{partition_type}#####")
    with open(os.path.join(PATH_DATA, f"{data_source}_{partition_type}_distance.csv"), "w") as f:
        header = ",".join(["label", "client", "x", "y"]) + "\n"
        f.write(header)
        for j_client in range(10):
            mdis = np.linalg.norm(gs_res[i_label][j_client]['mean'])
            vdis = np.linalg.norm(np.sqrt(gs_res[i_label][j_client]['var']))
            f.write("%d,%d,%.8f,%.8f\n" % (i_label, j_client, mdis, vdis))


if __name__ == '__main__':
    make_features("femnist", "homo")
    make_features("femnist", "real")

    distance_gs_res("femnist", "homo")
    distance_gs_res("femnist", "real")

    print("see result `*.csv` in data folder")