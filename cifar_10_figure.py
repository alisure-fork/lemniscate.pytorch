import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from alisuretool.Tools import Tools


class CosineDecayStrategy(object):

    @staticmethod
    def _adjust_learning_rate(learning_rate, epoch, t_epoch=100, first_epoch=200):

        def _get_lr(_base_lr, now_epoch, _t_epoch=t_epoch, _eta_min=1e-05):
            return _eta_min + (_base_lr - _eta_min) * (1 + math.cos(math.pi * now_epoch / _t_epoch)) / 2

        if epoch < first_epoch + t_epoch * 0:  # 0-200
            learning_rate = learning_rate
        elif epoch < first_epoch + t_epoch * 1:  # 200-300
            learning_rate = learning_rate / 2
        elif epoch < first_epoch + t_epoch * 2:  # 300-400
            learning_rate = learning_rate / 4
        elif epoch < first_epoch + t_epoch * 3:  # 400-500
            learning_rate = _get_lr(learning_rate / 2.0, epoch - first_epoch - t_epoch * 2)
        elif epoch < first_epoch + t_epoch * 4:  # 500-600
            learning_rate = _get_lr(learning_rate / 2.0, epoch - first_epoch - t_epoch * 3)
        elif epoch < first_epoch + t_epoch * 5:  # 600-700
            learning_rate = _get_lr(learning_rate / 4.0, epoch - first_epoch - t_epoch * 4)
        elif epoch < first_epoch + t_epoch * 6:  # 700-800
            learning_rate = _get_lr(learning_rate / 4.0, epoch - first_epoch - t_epoch * 5)
        elif epoch < first_epoch + t_epoch * 7:  # 800-900
            learning_rate = _get_lr(learning_rate / 8.0, epoch - first_epoch - t_epoch * 6)
        elif epoch < first_epoch + t_epoch * 8:  # 900-1000
            learning_rate = _get_lr(learning_rate / 8.0, epoch - first_epoch - t_epoch * 7)
        elif epoch < first_epoch + t_epoch * 9:  # 1000-1100
            learning_rate = _get_lr(learning_rate / 16., epoch - first_epoch - t_epoch * 8)
        elif epoch < first_epoch + t_epoch * 10:  # 1100-1200
            learning_rate = _get_lr(learning_rate / 16., epoch - first_epoch - t_epoch * 9)
        elif epoch < first_epoch + t_epoch * 11:  # 1200-1300
            learning_rate = _get_lr(learning_rate / 32., epoch - first_epoch - t_epoch * 10)
        elif epoch < first_epoch + t_epoch * 12:  # 1300-1400
            learning_rate = _get_lr(learning_rate / 32., epoch - first_epoch - t_epoch * 11)
        elif epoch < first_epoch + t_epoch * 13:  # 1400-1500
            learning_rate = _get_lr(learning_rate / 64., epoch - first_epoch - t_epoch * 12)
        else:  # 1500-1600
            learning_rate = _get_lr(learning_rate / 64., epoch - first_epoch - t_epoch * 13)
            pass

        return learning_rate

    @classmethod
    def line_lr(cls):
        epoch = np.array([i for i in range(1601)])
        lr = np.array([cls._adjust_learning_rate(0.01, i) for i in range(1601)])

        fig = plt.figure(figsize=(12, 4))
        plt.plot(epoch, lr)
        plt.xlabel('epoch')
        plt.ylabel('learning rate')
        plt.xticks([i * 100 for i in range(17)])
        plt.yticks([0.01 / 8 * i for i in range(9)])
        plt.grid(linestyle='--')
        # plt.show()

        plt.savefig("line_lr.png", dpi=300)
        pass

    pass


def show_dot():
    color = ["g", "r", "c", "m", "y", "k", "sienna", "orange", "lawngreen", "deepskyblue", "lightcoral", "b"]
    class_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig = plt.figure()
    for i, c in enumerate(color):
        if i < 10:
            # plt.text(0.4, 0.05 * i + 0.2, "● " + class_name[i], color=c, fontdict={'size': 10})

            # plt.text(0.4, 0.05 * i + 0.2, "● ", color=c, fontdict={'size': 10})
            # plt.text(0.435, 0.05 * i + 0.196, class_name[i], color="k", fontdict={'size': 10})

            plt.text(0.11 * i - 0.05, 0.2, "● ", color=c, fontdict={'size': 8})
            plt.text(0.11 * i + 0.03 - 0.05, 0.197, class_name[i], color="k", fontdict={'size': 8})
        pass

    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # plt.show()
    plt.savefig("dot.png", dpi=200)
    pass


def single_dot():
    color = ["g", "r", "c", "m", "y", "k", "sienna", "orange", "lawngreen", "deepskyblue"]
    class_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig = plt.figure()
    for i, c in enumerate(color):
        plt.text(0.08 * i, 0.5, "●", color=c, fontdict={'size': 8})

    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # plt.show()
    plt.savefig("single_dot.png", dpi=200)
    pass


def g0():
    cmaps = [('1 Sequential', [
                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
             ('2 Sequential (2)', [
                 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                 'hot', 'afmhot', 'gist_heat', 'copper']),
             ('3 Diverging', [
                 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
             ('4 Qualitative', [
                 'Pastel1', 'Pastel2', 'Paired', 'Accent',
                 'Dark2', 'Set1', 'Set2', 'Set3',
                 'tab10', 'tab20', 'tab20b', 'tab20c']),
             ('5 Miscellaneous', [
                 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, figh))
        fig.subplots_adjust(top=1 - .35 / figh, bottom=.15 / figh, left=0.2, right=0.99)

        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            ax.text(-.01, .5, name, va='center', ha='right', fontsize=10, transform=ax.transAxes)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()
        pass

    for cmap_category, cmap_list in cmaps:
        plot_color_gradients(cmap_category, cmap_list)
        pass

    plt.show()
    pass


def g():
    v1 = [np.random.rand(1)[0] for i in range(1024)]
    v2 = [np.random.rand(1)[0] for i in range(1024)]
    v3 = [np.random.rand(1)[0] for i in range(1024)]
    v4 = [np.random.rand(1)[0] for i in range(1024)]
    v5 = [np.random.rand(1)[0] for i in range(1024)]
    gradient = np.vstack((v1, v2, v3, v4, v5))

    fig, ax = plt.subplots()
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap("tab10"))
    ax.set_axis_off()
    plt.show()
    pass


def show_image():
    mv = 1024
    number = 1000
    data_all = [[np.random.randint(0, 64), np.random.randint(0, 128), np.random.randint(0, 256),
                 np.random.randint(0, 512), np.random.randint(0, 1024)] for i in range(number)]
    data_all = sorted(data_all, key=lambda x: x[0]*mv*mv*mv*mv + x[1]*mv*mv*mv + x[2]*mv*mv + x[3]*mv + x[4])
    color = ["g", "r", "c", "m", "y", "k", "sienna", "orange", "lawngreen", "deepskyblue"]
    class_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    figure, ax = plt.subplots()

    ax.set_xlim(left=0, right=number)
    ax.set_ylim(bottom=0, top=30)

    for data_index, data_one in enumerate(data_all):
        for i, data in enumerate(data_one):
            ax.add_line(Line2D((data_index, data_index), (15 + i, 15 + i + 1), linewidth=1, color=color[0]))
            pass
        pass

    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.plot()
    plt.show()
    # plt.savefig("dot.png", dpi=200)
    pass


def show_clustering():
    m = 1024
    pkl = Tools.read_from_pkl("class_clustering.pkl")
    data_all = [list(pkl["1"]/1024), list(pkl["2"]/512), list(pkl["3"]/256),
                list(pkl["4"]/128), list(pkl["5"]/64), list(pkl["0"])]

    data_all = np.transpose(data_all)
    data_all = sorted(data_all, key=lambda x: x[0]*m*m*m*m*m + x[1]*m*m*m*m + x[2]*m*m*m + x[3]*m*m + x[4]*m + x[5])
    # data_all = sorted(data_all, key=lambda x: x[5]*m*m*m*m*m + x[4]*m*m*m*m + x[3]*m*m*m + x[2]*m*m + x[1]*m + x[0])
    data_all = [data_one[0: -1] for data_one in data_all if data_one[-1] == 0]
    data_all = np.transpose(data_all)

    fig, ax = plt.subplots()
    ax.imshow(data_all, aspect='auto', cmap=plt.get_cmap("Purples"))
    ax.set_axis_off()
    plt.show()

    # plt.savefig("dot.png", dpi=200)
    pass


if __name__ == '__main__':
    # CosineDecayStrategy.line_lr()
    # show_dot()
    # single_dot()
    # g0()
    # g()
    # show_image()
    # show_clustering()
    pass
