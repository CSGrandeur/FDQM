# Test different level of noise in a dataset

import sys
import os
import numpy as np
PATH_BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PATH_BASE)
sys.path.append(os.path.join(PATH_BASE, 'demos'))
PATH_DATA = os.path.join(PATH_BASE, 'data')
import tools.dataset_util as du
from skimage import img_as_float
from methods import noise as ne

def test_noise():
    for noise_level in range(5, 100, 5):
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = du.partition_data(
            'fmnist', os.path.join(PATH_BASE, "data"), './logs/', 'iid-diff-quantity', 10, beta=1)

        aver_predict_sigma = []
        sigma = noise_level / 255
        for i in range(0, min(300, X_train.shape[0]), 4):
            im = np.concatenate(
                (np.concatenate((X_train[i], X_train[i + 1]), axis=0),
                np.concatenate((X_train[i + 2], X_train[i + 3]), axis=0)),
                axis=1
            )
            im = img_as_float(im)
            im_noise = im + np.random.randn(*im.shape) * sigma
            aver_predict_sigma.append(ne.noise_estimate(im_noise) * 255)

        print(noise_level, np.array(aver_predict_sigma).mean())


if __name__ == '__main__':
    test_noise()