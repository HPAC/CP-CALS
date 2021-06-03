import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from scripts.python.experiments_jk.data_reader import read_data
from scripts.python.utils.figure_config import fig_size_in, fig_format
from scripts.python.utils.system_config import plot_output_path
from scripts.python.utils.utils import modes_string, modes_title_string

if __name__ == '__main__':

    backend = 'MKL'

    modes_v = []
    modes_v.append((89, 97, 549))
    modes_v.append((44, 2700, 200))

    fig, ax = plt.subplots(2, 1, sharex='all')
    fig.set_size_inches(w=fig_size_in['width']*1.2, h=fig_size_in['height'] * 1.2)

    dict_v = []
    for i, modes in enumerate(modes_v):

        configurations = [(False, 1), (False, 24), (True, 24)]
        # configurations = [(False, 1), (False, 24)]
        df = pd.DataFrame({'Method': ['JK-ALS', 'JK-OALS', 'JK-CALS']})
        for (gpu, n_threads) in configurations:
            dict = read_data(backend, n_threads, modes, gpu)
            df = pd.merge(df, dict['jkdata'], on='Method')
            print(df.to_latex(float_format="{:0.2f}".format, na_rep='-'))

        indices = df['Method'].to_list()
        del df['Method']
        df.index = indices
        df = df.transpose()

        df = df[['JK-ALS', 'JK-OALS', 'JK-CALS']]
        colors = ['C0', 'C1', 'C2']
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
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom')
        ax[i].set_title(modes_title_string(modes))
        ax[i].set_ylabel('Time in seconds')
        print(plt.xticks())
    # title = 'Application problems'
    # fig.suptitle(title)
    ax[-1].set_xlabel('System Configurations')
    fig.tight_layout()
    plt.minorticks_off()

plt.savefig(plot_output_path
            + 'real'
            + fig_format)
plt.show()
