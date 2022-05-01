import os
import psutil
import sys
import shutil
import warnings
from inspect import currentframe

import json
import _pickle

import numpy as np

ENVS = ('PC', 'SERVER', 'COLAB', 'OTHER')


def name_function() -> str:
    return '---> ' + f'{currentframe().f_back.f_code.co_name:50}'


def give_mem():
    process = psutil.Process(os.getpid())
    mem = process.memory_percent()
    return np.around(mem, decimals=3)


def define_env():
    """
    Define if code is running on the notebook PC, Linux Server or Colab
    New options may be added later
    :return:
    """

    option = 'OTHER'
    if os.name == 'nt' and os.path.exists('C:\\Users\\Art'):
        option = 'PC'
    elif os.name != 'nt' and os.path.exists('/home/neo'):
        option = 'SERVER'
    elif os.name != 'nt' and os.path.exists('/content/drive/MyDrive/'):
        option = 'COLAB'

    assert option in ENVS
    return option


def get_project_path():
    project_path = ''
    if define_env() == 'SERVER':
        project_path = '/home/neo/PycharmProjects/Course1/'
    elif define_env() == 'PC':
        project_path = 'C:\\Users\\Art\\PycharmProjects\\Course1'
    elif define_env() == 'COLAB':
        project_path = '/content/drive/MyDrive/ColabNotebooks/Course1'
    return project_path


def check_dir_if_exists(dir_fp):
    if os.path.exists(dir_fp):
        if not os.path.isdir(dir_fp):
            warnings.warn(f'Path {dir_fp} is not a dir path or not a full path',
                          category=RuntimeWarning)
        return True
    return False


def check_file_if_exists(file_fp):
    if os.path.exists(file_fp):
        if not os.path.isfile(file_fp):
            warnings.warn(f'Path {file_fp} is not a file path or not a full path',
                          category=RuntimeWarning)
        return True
    return False


def create_empty_dir_unsafe(dir_fp):
    if os.path.exists(dir_fp) and os.path.isdir(dir_fp):
        if len(os.listdir(dir_fp)) != 0:
            shutil.rmtree(dir_fp)
        else:
            return
    os.mkdir(dir_fp)


def create_empty_dir_safe(dir_fp):
    if os.path.exists(dir_fp) and os.path.isdir(dir_fp):
        if len(os.listdir(dir_fp)) != 0:
            warnings.warn('Could not create a dir cause already exists',
                          category=RuntimeWarning)
            print('FAILED_TO_CREATE_DIR', dir_fp)
    else:
        os.mkdir(dir_fp)


def remove_dir_if_exists(dir_fp):
    # Checking that a dir_fp exists is not required
    if os.path.exists(dir_fp):
        shutil.rmtree(dir_fp)


def remove_file_if_exists(file_fp):
    # Checking that a file_fp exists is not required
    if os.path.exists(file_fp):
        os.remove(file_fp)


def read_json(file_fp):
    # Unsafe reading could be separated but omitted to restrict its usage
    if check_file_if_exists(file_fp):
        with open(file_fp, mode='r') as f:
            data = json.load(f)
    else:
        warnings.warn(f'Could not read JSON file {file_fp}',
                      category=RuntimeWarning)
        data = {}
    return data


def read_pkl(file_fp):
    # Unsafe reading could be separated but omitted to restrict its usage
    if check_file_if_exists(file_fp):
        with open(file_fp, mode='rb') as f:
            data = _pickle.load(f)
    else:
        warnings.warn(f'Could not read PKL file {file_fp}',
                      category=RuntimeWarning)
        data = {}
    return data


def read_np(file_fp):
    # Unsafe reading could be separated but omitted to restrict its usage
    if check_file_if_exists(file_fp):
        # Allow in order to load objects which contain numpy arrays
        data = np.load(file_fp, allow_pickle=True)
    else:
        warnings.warn(f'Could not read NP file {file_fp}',
                      category=RuntimeWarning)
        data = {}
    return data


def write_json_safe(file_fp, data):
    if check_file_if_exists(file_fp):
        print(f'JSON WRITE-SAFE FAIL {file_fp}')
        return
    with open(file_fp, mode='w') as f:
        json.dump(data, f)
    print(f'JSON WRITE-SAFE OK {file_fp}')


def write_json_unsafe(file_fp, data):
    remove_file_if_exists(file_fp)
    with open(file_fp, mode='w') as f:
        json.dump(data, f)
    print(f'JSON WRITE-UNSAFE OK {file_fp}')


def write_pkl_safe(file_fp, data):
    if check_file_if_exists(file_fp):
        print(f'PKL WRITE-SAFE FAIL {file_fp}')
        return
    with open(file_fp, mode='wb') as f:
        _pickle.dump(data, f)
    print(f'PKL WRITE-SAFE OK {file_fp}')


def write_pkl_unsafe(file_fp, data):
    remove_file_if_exists(file_fp)
    with open(file_fp, mode='wb') as f:
        _pickle.dump(data, f)
    print(f'PKL WRITE-UNSAFE OK {file_fp}')


def write_np_safe(file_fp, data):
    if check_file_if_exists(file_fp):
        print(f'PKL WRITE-SAFE FAIL {file_fp}')
        return
    np.save(file_fp, arr=data)
    print(f'NUMPY WRITE-SAFE OK {file_fp}')


def write_np_unsafe(file_fp, data):
    remove_file_if_exists(file_fp)
    np.save(file_fp, arr=data)
    print(f'NUMPY WRITE-UNSAFE OK {file_fp}')


# https://stackoverflow.com/a/67065084/8523656
def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda: None)
    return gettrace() is not None
