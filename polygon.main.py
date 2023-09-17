# Preprocess
import os

from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Basic Library
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# 1. Preprocessing including load data
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# 2. training and model
import time
import unet  # from unet.py
import torch.nn as nn
from torch import optim
from d2l import torch as d2l
from tqdm import tqdm
import pandas as pd


def check_cpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    """if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB') """

    return device


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
    polygons = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bounding_boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
        i = 0
        polygon = obj.find('polygon')
        po = []
        for point in polygon:
            i += 1
            if point.tag.startswith('x'):
                x = float(point.text)
            elif point.tag.startswith('y'):
                y = float(point.text)
                po.append([x, y])

        polygons.append(po)

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    return bounding_boxes, labels, width, height, polygons


def display_image_with_boxes(image):
    bboxes, labels, width, height, polygons = parse_xml(self.anno_idx[item])
    img_with_boxes = image.clone().permute(1, 2, 0).numpy()
    plt.imshow(img_with_boxes)
    ax = plt.gca()
    for bbox in boxes:
        xmin, ymin, xmax, ymax = bbox
        color = 'blue' if lbls == 0 else 'red'  # blus as leaf , red as stem
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
    plt.show()


def func(batch):
    img, label = zip(*batch)
    img_list = []
    label_list = []
    for i, l in enumerate(label):
        img_ptr = torch.full((l.size(0), 1), i, dtype=torch.int32)
        l = torch.cat((img_ptr, l), dim=1)
        return [torch.stack(img,0), torch.cat(label, 0)]




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
        self.count = 0
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

        res = []  # Store annotation information, i.e., coordinates of the bounding box's top left and bottom right
        # points and the target's class label
        result = []
        la = []

        if os.path.exists(self.root_path):
            bboxes, labels, width, height, polygons = parse_xml(self.anno_idx[item])

            lbls = convert_label(labels)  # Convert label to number using convert_label() function

            # res.append(bboxes)
            res.append(polygons)
            res.append(lbls)

            for polygons, label in zip(*res):
                polygons_with_label = [polygons + [label] for polygons in polygons]
                result.extend(polygons_with_label)

        else:
            raise Exception('Path does not Exist!')

        la.append(lbls)
        result = torch.from_numpy(np.array(result))
        # print(result.size())  # check how many label of one images  e.g [239,3]

        return img, result

    def __len__(self):
        return len(self.img_idx)


import torch.nn.functional as F


# Train Part
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Move data to the same device as the model
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        

        # Show the image using matplotlib
        plt.imshow(pred_pil)
        plt.axis('off')
        plt.show()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # Move data to the same device as the model
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            pred = F.interpolate(pred, size=(320, 320), mode='bilinear', align_corners=False)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    root_dir = "C:/Users/willi/PycharmProjects/modelling/Dataset/test/"
    train_data = Read_voc(root_path=root_dir)  # DataSet Preprocessing
    device = check_cpu()  # checking the training module is using cpu/gpu?
    # display_image_with_boxes(image)
    """img, label = train_data[0]
    print(img.size())
    print(label.size())
    print(train_data.__len__())
    display_image_with_boxes(img, res, label)"""

    """train_dataset, test_dataset = random_split(
        dataset=train_data,
        lengths=[2, 0],
        generator=torch.Generator().manual_seed(0)
    )"""

    # Display image and label.
    size = 5
    train_dataloader = DataLoader(train_data, batch_size=size, shuffle=True, collate_fn=func)
    # test_dataloader = DataLoader(test_dataset, batch_size=size, shuffle=True, collate_fn=func)
    count = 0
    # Display image and label.
    for index, data in enumerate(train_dataloader):
        features, labels = data
        print(f"Feature batch shape: {features.size()}")
        print(f"Labels batch shape: {labels.size()}")
        count = count + 1
    print(count)
    # print(len(train_dataloader))         #check the integrity of dataset loader
    # print(len(test_dataloader))

    # Model Configuration
    """model = unet.UNet(n_channels=3, n_classes=1 + 3, bilinear=True).to(device)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=1e-5, weight_decay=1e-8, momentum=0.999)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    loss_fn = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    epochs_num = 5
    # Train start
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    # train(model, train_dataloader, test_dataloader, criterion, optimizer, epochs_num, scheduler, device) """

    CLASSES = ['stem', 'soil', 'leaf']

    model = unet.UNet(n_channels=3, n_classes=len(CLASSES))
    model = model.cuda()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    epochs = 5
    for t in range(epochs):
        start_time = time.time()
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        #test_loop(test_dataloader, model, loss_fn, device)
    end_time = time.time()


main()
