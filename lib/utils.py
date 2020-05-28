import os
import shutil
import numpy as np

def get_files(path):
    return np.array([os.path.join(path, f) for f in os.listdir(path) 
                     if os.path.isfile(os.path.join(path, f))])

def make_dirs(path, remake=False):
    if path != None:
        if os.path.exists(path):
            if remake:
                shutil.rmtree(path)
                os.makedirs(path)
        else:
            os.makedirs(path)

def basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def find_paths_contains(name, paths):
    result = np.empty(0)
    for path in paths:
        if name in path:
            result = np.append(result, path)
    return result

def get_clean_in_mix(mix_filename, noise_prefix='n:'):
    return np.array([s for s in mix_filename.split(".") 
        if ":" in s and not s.startswith(noise_prefix)])

