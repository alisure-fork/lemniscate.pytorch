import os
from glob import glob
from PIL import Image
from collections import Counter


class Config(object):
    vis_root = "/mnt/4T/ALISURE/Unsupervised/vis/stl10_3"

    split = "unlabeled"
    # split = "train"
    # split = "test"

    ic_id_list = [4, 5, 29, 8, 41, 12, 45, 46, 64, 65]
    image_size = 84

    image_num_unlabeled = 18
    image_num_train = 18
    image_num_test = 18

    margin_image = 4
    margin_split = 32

    sort = True
    # sort = False

    if split == "train":
        image_num = image_num_train
        vis_ic_path = os.path.join(vis_root, split)
        result_size = (image_size * image_num_train + (image_num_train - 1) * margin_image,
                       len(ic_id_list) * image_size + (len(ic_id_list) - 1) * margin_image)
    elif split == "unlabeled":
        image_num = image_num_unlabeled
        vis_ic_path = os.path.join(vis_root, split)
        result_size = (image_size * image_num_unlabeled + (image_num_unlabeled - 1) * margin_image,
                       len(ic_id_list) * image_size + (len(ic_id_list) - 1) * margin_image)
    elif split == "test":
        image_num = image_num_test
        vis_ic_path = os.path.join(vis_root, split)
        result_size = (image_size * image_num_test + (image_num_test - 1) * margin_image,
                       len(ic_id_list) * image_size + (len(ic_id_list) - 1) * margin_image)
    else:
        raise Exception(".......")

    result_path = os.path.join(vis_root, "{}_{}_{}_{}.png".format(split, image_size, image_num, margin_image))
    pass


def get_image_y_split(vis_ic_path, ic_id, image_num):
    ic_image_file = glob(os.path.join(vis_ic_path, str(ic_id), "*.png"))

    if Config.sort:
        ic_image_file = sorted(ic_image_file)[::-1]
        pass

    im_list_result = [Image.open(image_file) for image_file in ic_image_file[:image_num]]
    return im_list_result


if __name__ == '__main__':
    im_result = Image.new("RGB", size=Config.result_size, color=(255, 255, 255))
    for i in range(len(Config.ic_id_list)):
        im_list = get_image_y_split(Config.vis_ic_path, ic_id=Config.ic_id_list[i], image_num=Config.image_num)
        for j in range(len(im_list)):
            im_result.paste(im_list[j].resize((Config.image_size, Config.image_size)),
                            box=(j * (Config.image_size + Config.margin_image),
                                 i * (Config.image_size + Config.margin_image)))
            pass
        pass
    im_result.save(Config.result_path)
    pass
