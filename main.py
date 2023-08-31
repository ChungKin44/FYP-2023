# Preprocess
import os
import cv2
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img

# Machine Learning
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


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


def normalize(img_numpy_array):  # gray scale
    return img_numpy_array / 255.0


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bounding_boxes = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bounding_boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    return bounding_boxes, labels, width, height


""" def write_folder_names_to_text(folder_path, output_file):   #list all name of jpg into txt 
    # Get the list of file names in the folder
    file_names = os.listdir(folder_path)

    # Create or overwrite the output file
    with open(output_file, 'w') as f:
        # Write each file name to a new line in the text file
        for file_name in file_names:
            if file_name.endswith('.jpg'):
                f.write(file_name + '\n')

    print(f"File names written to {output_file} successfully.")


folder_path = "C:/Users/willi/OneDrive/桌面/Dataset/test"
output_file = "C:/Users/willi/OneDrive/桌面/Dataset/test/file_names.txt"
write_folder_names_to_text(folder_path, output_file) """

""" def xml_to_csv(root_dir):                      #XML to CSV format 
    bbox = []
    labels = []
    for xml_file in glob.glob(root_dir + '/*.xml'):
        if os.path.exists(xml_file):
            bboxes, lbls, width, height = parse_xml(xml_file)
            bbox.append(bboxes)
            labels.append(lbls)
            label_in_num = convert_label(lbls)

    print(len(bbox))
    print(len(labels))
    csv_file = os.path.join(root_dir, 'annotations.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])

        for i in range(len(bbox)):
            for j in range(len(bbox[i])):
                writer.writerow([os.path.basename(xml_file), width, height, labels[i][j], bbox[i][j][0], bbox[i][j][1],
                                 bbox[i][j][2], bbox[i][j][3]])

    print(f"CSV file saved at: {csv_file}")  """


def display_image_with_boxes(image, boxes):
    img_with_boxes = image.clone().permute(1, 2, 0).numpy()
    plt.imshow(img_with_boxes)
    ax = plt.gca()
    for bbox in boxes:
        xmin, ymin, xmax, ymax, lbls = bbox
        color = 'blue' if lbls == 0 else 'red'  # blus as leaf , red as stem
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
    plt.show()


voc_label = {'leaf,stem,soil'}
dict_labels = dict(zip(voc_label, range(len(voc_label))))


class Read_voc(Dataset):
    def __init__(self, root_path):
        super(Read_voc, self).__init__()
        self.root_path = root_path
        self.img_idx = []
        self.anno_idx = []
        self.bbox = []
        self.obj_name = []
        train_txt_path = self.root_path + "/file_names.txt"
        self.img_path = self.root_path
        self.anno_path = self.root_path

        train_txt = open(train_txt_path)
        lines = train_txt.readlines()
        for line in lines:
            name = line.strip().split()[0]
            name = name.rstrip('.jpg')
            self.img_idx.append(self.img_path + name + '.jpg')
            self.anno_idx.append(self.anno_path + name + '.xml')

    def __getitem__(self, item):
        img = Image.open(self.img_idx[item])
        img = transforms.ToTensor()(img)
        normalize(img)
        targets = ET.parse(self.anno_idx[item])
        res = []  # Store annotation information, i.e., coordinates of the bounding box's top left and bottom right
        # points and the target's class label
        result = []
        if os.path.exists(self.root_path):
            bboxes, labels, width, height = parse_xml(self.anno_idx[item])
            lbls = convert_label(labels)  # Convert label to number using convert_label() function
            res.append(bboxes)
            res.append(lbls)
            for bbox, label in zip(*res):
                result.append(bbox + [label])

        else:
            raise Exception('Path does not Exist!')
        result = torch.stack([torch.tensor(box) for box in result])
        # Convert result to a PyTorch tensor using torch.stack()
        return img, result

    def __len__(self):
        return len(self.img_idx)


def main():
    root_dir = "C:/Users/willi/OneDrive/桌面/Dataset/test/"
    image_size = (256, 256)
    train_data = Read_voc(root_path=root_dir)  # DataSet Preprocessing
    img, res = train_data[0]
    print(img.size())
    print(len(res))  # row
    print(len(res[0]))  # cols
    print(train_data.__len__())
    display_image_with_boxes(img, res)
    # Display image and label.
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    train_features, train_labels = train_dataloader
    print(train_features.size())


main()

"""
# Network of CNN

class Network(nn.Module):

    def __init__(self, num_classes=21, groups=2):
        super(Network, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0))
        self.conv.add_module('relu1_s1', nn.ReLU(inplace=True))
        # self.conv.add_module('bn1_s1',nn.BatchNorm2d(96))
        self.conv.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=groups))
        self.conv.add_module('relu2_s1', nn.ReLU(inplace=True))
        # self.conv.add_module('bn2_s1',nn.BatchNorm2d(256))
        self.conv.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3_s1', nn.ReLU(inplace=True))
        # self.conv.add_module('bn3_s1',nn.BatchNorm2d(384))

        self.conv.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=groups))
        # self.conv.add_module('bn4_s1',nn.BatchNorm2d(384))
        self.conv.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=groups))
        # self.conv.add_module('bn5_s1',nn.BatchNorm2d(256))
        self.conv.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Linear(256 * 6 * 6, 4096))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1', nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7', nn.Linear(4096, 4096))
        self.fc7.add_module('relu7', nn.ReLU(inplace=True))
        self.fc7.add_module('drop7', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8', nn.Linear(4096, num_classes))   """
