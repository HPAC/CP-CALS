import logging
import os.path

import pandas as pd

from scripts.python.utils.system_config import input_path
from scripts.python.utils.utils import modes_string

logger = logging.getLogger('Data reader')


def read_data(backend, threads, modes, gpu=False):
    results_dict = {}
    folder = input_path + '{}/'.format(backend)
    mode_name = modes_string(modes)

    if gpu:
        jk_file_name = '{}JKCUDA__{}_{}.csv'.format(folder, mode_name, threads)
    else:
        jk_file_name = '{}JK_{}_{}.csv'.format(folder, mode_name, threads)

    if not os.path.isfile(jk_file_name):
        logger.warning('File: {} not found.'.format(jk_file_name))
        jkdata = []
    else:
        jkdata = pd.read_csv(jk_file_name, delimiter=';')

    results_dict.update({'backend': backend,
                         'threads': threads,
                         'modes': modes,
                         'gpu': gpu,
                         'jkdata': jkdata})
    return results_dict