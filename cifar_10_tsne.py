import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from alisuretool.Tools import Tools
from cifar_10_tool import FeatureName


class Cifar10TSNE(object):

    def t_sne(self, feature_file, feature_name, result_file, result_png, s=None, reset=True):
        Tools.print('Computing t-SNE embedding: {} {}'.format(feature_name, feature_file))

        if reset or not os.path.exists(result_file):
            Tools.print('read t-SNE embedding data')
            feature, label = self.load_feature(feature_file, feature_name)
            feature = np.reshape(feature, (len(feature), -1))
            # feature, label = feature[: 100], label[: 100]
            t_sne = TSNE(n_components=2)
            embedding = t_sne.fit_transform(feature)
            Tools.write_to_pkl(result_file, {"label": label, "embedding": embedding})
        else:
            Tools.print('exists t-SNE embedding data: {}'.format(result_file))
            embedding_dict = Tools.read_from_pkl(result_file)
            label = embedding_dict["label"]
            embedding = embedding_dict["embedding"]
            pass

        Tools.print("begin to plot embedding")
        self.plot_embedding(embedding, label, result_png, s)
        pass

    @staticmethod
    def load_feature(feature_file, feature_name):
        feature_dict = Tools.read_from_pkl(feature_file)
        feature = feature_dict[FeatureName.feature][feature_name]
        label = feature_dict[FeatureName.label]
        return feature, label

    @staticmethod
    def plot_embedding(data, label, result_png, s=None):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        # data = np.asarray([a for a in data if a[0] < 0.8 and a[1] > 0.1])  # x
        data = np.asarray([a for a in data if a[0] < 0.4 and a[1] > 0.5])  # ConvB3
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        color = ["g", "r", "c", "m", "y", "k", "sienna", "orange", "lawngreen", "deepskyblue", "lightcoral", "b"]
        fig = plt.figure()
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], s if s is not None else str(label[i]),
                     color=color[label[i]], fontdict={'weight': 'bold', 'size': 2})
            pass

        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

        Tools.print("begin to save {}".format(result_png))
        plt.savefig(result_png, dpi=200)
        return fig

    pass


if __name__ == '__main__':
    _checkpoint_path = "11_class_1024_5level_512_256_128_64_no_1600_32_1_l1_sum_0_54321"

    is_train = False
    _feature_name = FeatureName.ConvB3

    _feature_file = "./checkpoint/{}/feature_{}.pkl".format(_checkpoint_path, "train" if is_train else "test")
    _result_png = "./checkpoint/{}/feature/feature_{}_{}.png".format(
        _checkpoint_path, "train" if is_train else "test", _feature_name)
    _result_file = "./checkpoint/{}/feature/feature_{}_{}.pkl".format(
        _checkpoint_path, "train" if is_train else "test", _feature_name)

    Tools.print("{}".format(_result_png))
    cifar_t_sne = Cifar10TSNE()
    cifar_t_sne.t_sne(_feature_file, _feature_name, _result_file, _result_png, s="●", reset=False)
    pass
