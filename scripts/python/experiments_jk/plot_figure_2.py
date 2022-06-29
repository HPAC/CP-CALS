import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from scripts.python.experiments_jk.data_reader import read_data
from scripts.python.utils.figure_config import fig_size_in, fig_format
from scripts.python.utils.system_config import plot_output_path
from scripts.python.utils.utils import modes_string, modes_title_string

if __name__ == '__main__':

    configurations = [(False, 1), (False, 24), (True, 24)]
    for (gpu, n_threads) in configurations:

        modes_v = []
        modes_v.append((50, 100, 100))
        modes_v.append((50, 200, 200))
        modes_v.append((50, 400, 400))

        backend = 'MKL'

        fig, ax = plt.subplots(3, 1, sharex='all')
        fig.set_size_inches(w=fig_size_in['width']*1.15, h=fig_size_in['height']*1.9)

        dict_v = []
        for i, modes in enumerate(modes_v):
            dict = read_data(backend, n_threads, modes, gpu)

            df = dict['jkdata']

            indices = df['Method'].to_list()
            del df['Method']
            df.index = indices
            df = df.transpose()
            indices = df.index.to_list()
            indices[-1] = 'All'
            df.index = indices
            df.at['All', 'JK-ALS'] = df['JK-ALS'].sum()

            colors = ['C0', 'C2', 'C1']
            if n_threads == 24:
                df = df[['JK-ALS', 'JK-OALS', 'JK-CALS']]
                colors = ['C0', 'C1', 'C2']
            else:
                del df['JK-OALS']
            legend = False
            if i == 1:
                legend = True
            df.plot.bar(ax=ax[i], color=colors, rot=0, legend=legend)

            old_lim = list(ax[i].get_ylim())
            old_lim[1] += 0.1 * old_lim[1]
            ax[i].set_ylim(old_lim)
            for p in ax[i].patches:
                if p.get_height() < 5:
                    height = round(p.get_height(), 1)
                else:
                    height = int(round(p.get_height()))
                if not p.get_height() == 0:
                    ax[i].annotate(str(height),
                                   xy=(p.get_x() + p.get_width() / 2, height),
                                   xytext=(0, 2),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
            ax[i].set_title(modes_title_string(modes))
            ax[i].set_ylabel('Time in seconds')
        title = 'Single Threaded Execution'
        if n_threads == 24:
            if gpu:
                title = 'Offloading of the MTTKRP to the GPU'
            else:
                title = 'Multi Threaded Execution (24 threads)'
        fig.suptitle(title)
        ax[-1].set_xlabel('Components')
        fig.tight_layout()
        specifier = str(n_threads)
        if gpu:
            specifier = 'GPU'
        plt.savefig(plot_output_path
                    + 'artif_'
                    + specifier
                    + fig_format)

    plt.show()
