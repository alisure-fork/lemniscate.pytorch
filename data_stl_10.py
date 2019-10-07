import os
import numpy as np
from imageio import imsave
from alisuretool.Tools import Tools


class STL10(object):

    def __init__(self, height=96, width=96, depth=3, data_root="/mnt/4T/Data/STL10/stl10_binary",
                 result_path="/mnt/4T/Data/STL10/stl10_img"):
        self.height = height
        self.width = width
        self.depth = depth

        self.size = self.height * self.width * self.depth
        self.data_root = data_root
        self.result_path = Tools.new_dir(result_path)

        self.train_unlabel_path = os.path.join(self.data_root, "unlabeled_X.bin")
        self.train_x_path = os.path.join(self.data_root, "train_X.bin")
        self.train_y_path = os.path.join(self.data_root, "train_y.bin")
        self.test_x_path = os.path.join(self.data_root, "test_X.bin")
        self.test_y_path = os.path.join(self.data_root, "test_y.bin")
        pass

    @staticmethod
    def read_file(file_path):
        with open(file_path, 'rb') as _f:
            return np.fromfile(_f, dtype=np.uint8)
        pass

    def read_images(self, path_to_data):
        everything = self.read_file(path_to_data)
        return np.transpose(np.reshape(everything, (-1, self.depth, self.height, self.width)), (0, 3, 2, 1))

    def read_labels(self, path_to_data):
        return self.read_file(path_to_data)

    def save_images(self, images, labels=None, is_train=False):
        Tools.print("Saving images to disk")
        for index, image in enumerate(images):
            name = "unlabel" if labels is None else ("train" if is_train else "test")
            name = name if labels is None else "{}/{}".format(name, labels[index])
            result_path = Tools.new_dir("{}/{}/{}.png".format(self.result_path, name, index))

            imsave(result_path, image, format="png")
            if index % 1000 == 0:
                Tools.print("{} {} {}".format(index, len(images), result_path))
            pass
        pass

    pass


if __name__ == "__main__":
    stl_10 = STL10()

    _images = stl_10.read_images(stl_10.test_x_path)
    _labels = stl_10.read_labels(stl_10.test_y_path)
    print(_images.shape)
    print(_labels.shape)
    stl_10.save_images(_images, _labels, is_train=False)

    _images = stl_10.read_images(stl_10.train_x_path)
    _labels = stl_10.read_labels(stl_10.train_y_path)
    print(_images.shape)
    print(_labels.shape)
    stl_10.save_images(_images, _labels, is_train=True)

    _images = stl_10.read_images(stl_10.train_unlabel_path)
    print(_images.shape)
    stl_10.save_images(_images)
    pass
