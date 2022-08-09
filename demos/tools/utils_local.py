import os
import pickle

def save_pickle(python_var, path_pickle):
    pickle.dump(python_var, open(path_pickle, "wb"))

def load_pickle(path_pickle):
    return pickle.load(open(path_pickle, "rb"))

def mkdir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except Exception as e:
            return repr(e)
    return None

from skimage import io
def save_image(path, img):
    io.imsave(path, img)