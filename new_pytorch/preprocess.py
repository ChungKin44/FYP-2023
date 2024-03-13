# Preprocess
import os
# Basic Library
import xml.etree.ElementTree as ET

import cv2
import numpy as np
# 1. Preprocessing including load data
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

label_class = [0, 1]  # label only lead and stem
import torch.nn.functional as F


def normalize(img_numpy_array):  # gray scale
    return img_numpy_array / 255.0


def convert_label(lab):
    result = []
    for str in lab:
        if str == 'leaf':
            result.append(0)
        elif str == 'stem':
            result.append(1)
        elif str == 'soil':
            result.append(2)
    return result


def polygon_to_mask(labels):
    try:
        # Create a black background
        img = np.zeros((480, 640, 3), dtype=np.uint8)  # initial step for making mask images

        for polygon_info in labels:
            polygon_num, polygon_coords, label = polygon_info

            # Convert to integer coordinates
            polygon_coords = [[int(x), int(y)] for x, y in polygon_coords]

            if label == 0:
                face_color = [0., 1., 0.]  # Green  onehot encode with label classes
            elif label == 1:
                face_color = [0., 0., 1.]  # Blue
            else:
                face_color = [1., 0., 0.]  # Red  others

            # Fill the polygon with its color in the image
            cv2.fillPoly(img, [np.array(polygon_coords)], color=face_color)

        mask_array = np.array(img)
        mask_array[(mask_array == [0, 0, 0]).all(axis=2)] = [1., 0., 0.]

        return mask_array
    except Exception as e:
        print(e)
    finally:
        labels.clear()


def parse_xml(xml_path):
    # print(xml_path)
    results = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bounding_boxes = []
    labels = []
    polygons = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        labels.append(name)
        polygon = obj.find('polygon')
        po = []
        for point in polygon:
            count = 0
            if point.tag.startswith('x'):
                x = float(point.text)
            elif point.tag.startswith('y'):
                y = float(point.text)
                po.append([x, y])
        polygons.append(po)
    return labels, polygons


def count_classes(one_hot):
    red_pixels = np.sum(np.all(one_hot == [0, 0, 1], axis=-1))
    green_pixels = np.sum(np.all(one_hot == [0, 1, 0], axis=-1))
    blue_pixels = np.sum(np.all(one_hot == [1, 0, 0], axis=-1))
    print("Red pixels count:", red_pixels)
    print("Green pixels count:", green_pixels)
    print("Blue pixels count:", blue_pixels)

    return 0


def to_OneHot(mask, size):
    mask = cv2.resize(mask, size)

    red = np.array([0, 0, 255])
    green = np.array([0, 255, 0])
    one_hot = np.zeros_like(mask)
    green_mask = np.all(mask == green, axis=-1)
    red_mask = np.all(mask == red, axis=-1)
    blue_mask = ~(red_mask | green_mask)
    one_hot[red_mask] = np.array([0, 0, 1])  # stem
    one_hot[green_mask] = np.array([0, 1, 0])  # leaf
    one_hot[blue_mask] = np.array([1, 0, 0])  # background

    # count_classes(one_hot)
    return one_hot


class Read_voc(Dataset):
    def __init__(self, root_path):
        super(Read_voc, self).__init__()
        self.root_path = root_path
        self.img_idx = []
        self.anno_idx = []
        self.obj_name = []
        train_txt_path = self.root_path + "/file_names.txt"
        self.img_path = self.root_path
        self.anno_path = self.root_path
        self.count = 0
        train_txt = open(train_txt_path)
        lines = train_txt.readlines()
        for line in lines:
            name = line.strip().split()[0]
            name = name.rstrip('.jpg')
            self.img_idx.append(self.img_path + name + '.jpg')
            self.anno_idx.append(self.anno_path + name + '.xml')

    def __getitem__(self, item):
        og_size = (320, 240)
        # Read image

        img = cv2.imread(self.img_idx[item])
        img = cv2.resize(img, og_size)
        img = torch.from_numpy(img).float()

        # Read Mask created already
        img_idx = self.img_idx[item]
        mask_path = img_idx.replace("./Dataset/Ash/augmented/", "./Dataset/Ash/augmented_mask/")
        mask_path = os.path.splitext(mask_path)[0] + ".png"
        mask = cv2.imread(mask_path)
        mask = to_OneHot(mask, og_size)
        mask = torch.from_numpy(mask).float()

        # change dimension
        img = normalize(img)
        mask = mask.permute(2, 0, 1)
        img = img.permute(2, 0, 1)
        return img, mask  # must be img shape: torch.Size([3, 480, 640]) Labels shape: torch.Size([232, 3])

    def __len__(self):
        return len(self.img_idx)

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# def main():
#     root_path = r'./Dataset/Ash/augmented/'
#     train_data = Read_voc(root_path=root_path)
#     # train_features, train_labels = next(iter(train_data))
#     # print(f"Feature batch shape: {train_features.size()}")
#     # print(f"Labels batch shape: {train_labels.size()}")
#
#     batch_size = 2
#     count = 0
#     train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     for index, data in enumerate(train_dataloader):
#         features, labels = data
#         print(f"Feature batch shape: {features.size()}")
#         print(f"Labels batch shape: {labels.size()}")
#         count = count + 1
#     print(count)
#
#     # # Display image and label.
#     # train_features, train_labels = next(iter(train_dataloader))
#     # print(f"Feature batch shape: {train_features.size()}")
#     # print(f"Labels batch shape: {train_labels.size()}")
#     # img = train_features[0].squeeze()
#     # img = img.permute(1, 2, 0) * 255
#     # label = train_labels[0].squeeze()
#     # label = label.permute(1, 2, 0) * 255
#     # # plt.imshow(img)
#     # # plt.show()
#     # plt.imshow(label)
#     # plt.show()
#
# #
# main()
