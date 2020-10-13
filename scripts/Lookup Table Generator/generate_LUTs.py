from system_config import *
from utils import *

import pandas as pd
import os
import os.path
from pathlib import Path


def read_bench(backend, threads, modes, mttkrp_method):
    bench_dict = {}
    folder = input_path + '{}/benchmark/'.format(backend)
    mode_name = mode_string(modes)
    bench_file_name = '{}benchmark_{}_{}_{}_{}.csv'.format(folder, backend, mode_name, threads, mttkrp_method)
    if os.path.isfile(bench_file_name):
        print('Found file: ', bench_file_name)
        bench_dict.update({'backend': backend,
                           'modes': modes,
                           'threads': threads,
                           'mttkrp_method': mttkrp_method,
                           'data': pd.read_csv(bench_file_name, delimiter=';')})
    return bench_dict


def best_method_per_rank(backend, modes, threads):
    mttkrp_methods = [MttkrpMethod.MTTKRP, MttkrpMethod.TWOSTEP0, MttkrpMethod.TWOSTEP1]
    best = []

    for id, method in enumerate(mttkrp_methods):
        get_best_method(best, read_bench(backend, threads, modes, method), method)

    export_to_file(backend, modes, threads, best)


def export_to_file(backend, modes, threads, best):
    path_to_files = lut_output_path + backend + '/lookup_tables/' + mode_string(modes) + '/' + str(threads)
    print('Path to files: {}'.format(path_to_files))

    Path(path_to_files).mkdir(parents=True, exist_ok=True)

    for idx, mode in enumerate(modes):
        f = open(path_to_files + '/' + str(idx), "w")
        for entry in best[idx]:
            f.write('{} {}\n'.format(entry[0], entry[3]))


def get_best_method(best, bench_dict, method):
    dic = bench_dict

    for midx, mode in enumerate(dic['modes']):

        df = dic['data']
        dm = df.loc[df['MODE'] == midx]
        if len(best) < midx + 1:
            best.append([(e['RANK'], e['FLOPS'], e['TIME'], int(method)) for i, e in dm.iterrows()])
        else:
            for i, e in dm.reset_index().iterrows():
                if best[midx][i][1] / best[midx][i][2] < e['FLOPS'] / e['TIME']:
                    best[midx][i] = (e['RANK'], e['FLOPS'], e['TIME'], int(method))


if __name__ == '__main__':

    backend = 'MKL'
    for modes in [(100, 100, 100), (200, 200, 200), (300, 300, 300)]:
        for threads in [1, 12, 24]:
            best_method_per_rank(backend, modes, threads)
