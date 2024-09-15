from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import utils


class Color:
    io_color = '#c55a11'
    comp_color = '#70ad47'
    comp_color = '#70ad47'
    forward_color = '#A9D18E'
    backward_color = '#4d7d2e'
    compression_color = '#4672C4'
    comm_color = '#F1B183'
    synceasgd_color = '#3F5EBA'
    opt_comm_color = '#c55a11'
    lars_color = '#A9D18E'


def plot_breakdown():
    FONTSIZE=16
    names = ['FF & BP', 'GradComm', 'FactorComp', 'FactorComm', 'InverseComp', 'InverseComm']
    colors = [Color.backward_color, Color.comm_color, Color.lars_color, Color.io_color, Color.compression_color, Color.synceasgd_color]
    sgd = [0.132,    0, 0,    0, 0, 0]
    ssgd = [0.132,    0.067, 0,    0, 0, 0]
    kfac = [0.132,    0, 0.205, 0,    0, 0.15, 0]
    dkfac = [0.132,    0.199-0.132, 0.404-0.199, 0.704-0.404, 0.736-0.704, 0.882-0.736]
    #names = ['FF & BP', 'Compression', 'Communication', 'LARS']
    #colors = [Color.backward_color, Color.compression_color, Color.comm_color, Color.lars_color]
    #densesgd = [0.204473, 0,    0.24177, 0.01114]
    #topksgd =  [0.204473,   0.239, 0.035, 0.01114]
    #densesgd96 = [0.054376, 0,    0.366886, 0.012794]
    #topksgd96 =  [0.054376,   0.239, 0.035, 0.012794]


    fig, ax = plt.subplots(figsize=(4.8,4.4))

    count = 2
    ind = np.arange(count)
    width = 0.28
    margin = 0.02
    xticklabels = ['SGD', 'KFAC']
    newind = np.arange(count).astype(np.float32)
    bars = []
    bottom=np.array([0,0]).astype(np.float32)
    for i in range(len(sgd)):
        label = names[i]
        data=[sgd[i], kfac[i]]
        p1 = ax.bar(newind, data, width, bottom=bottom, color=colors[i], label=label, edgecolor='black')
        bottom += np.array(data)
        bars.append(p1[0])
    utils.autolabel(p1, ax, r'1 GPU', 0, 10)

    newind += width+margin
    bottom=0
    for i in range(len(sgd)):
        label = names[i]
        data=[ssgd[i], dkfac[i]]
        p1 = ax.bar(newind, data, width, bottom=bottom, color=colors[i], label=label, edgecolor='black')
        bottom += np.array(data)
    utils.autolabel(p1, ax, r'64 GPUs', 0, 10)
        #bars.append(p1[0])


    handles, labels = ax.get_legend_handles_labels()
    #ax.legend([handles[0][0]], [labels[0][0]], ncol=2)
    print(labels)
    print(handles)
    #ax.set_xlim(right=2.5)
    ax.set_ylim(top=ax.get_ylim()[1]*1.05)
    ax.set_xticks(newind-(width+margin)/2)
    ax.set_xticklabels(xticklabels)
    #ax.set_xlabel('Model')
    ax.set_ylabel('Time [s]')
    utils.update_fontsize(ax, FONTSIZE)
    ax.legend(tuple(bars), tuple(names), loc='center left',bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE)#, handletextpad=0.2, columnspacing =1.)
    #fig.subplots_adjust(left=0.16, right=0.96, bottom=0.19, top=0.94)
    #plt.savefig('%s/naive-breakdown.pdf' % (OUTPUT_PATH), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_breakdown()
