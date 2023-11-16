# Preprocess
import os

from torch.optim.lr_scheduler import ExponentialLR, StepLR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision.utils import save_image

# Basic Library
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# 1. Preprocessing including load data
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from matplotlib.patches import Polygon
import matplotlib

import matplotlib.pyplot as plt

lbl_size_tracker = []

# 2. training and model
import time
import unet  # from unet.py
import torch.nn as nn
from torch import optim
from d2l import torch as d2l
from tqdm import tqdm
import pandas as pd

# 2.1 training parameter
from sklearn.metrics import f1_score, accuracy_score, jaccard_score


def check_cpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Additional Info when using cuda
    """if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB') """

    return device


def show_Unet_result(predict):
    predict = predict.cpu()
    output_image = predict[0].detach()
    output_image = np.array(output_image)
    output_image = denormalize(output_image)
    output_image = [[[output_image[i][j][k] for i in range(len(output_image))]
                     for j in range(len(output_image[0]))]
                    for k in range(len(output_image[0][0]))]

    output_image = np.clip(output_image, 0, 1)

    # Display the image
    plt.imshow(output_image)
    plt.title('After Unet')
    plt.axis('off')
    plt.show()

    return 0


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


def denormalize(img_3d_list):
    modified_image = []

    for row in img_3d_list:
        modified_row = []
        for col in row:
            modified_col = []
            for pixel in col:
                modified_pixel = pixel * -1 if pixel < 0 else pixel * 1
                modified_col.append(modified_pixel)
            modified_row.append(modified_col)
        modified_image.append(modified_row)

    return modified_image


def display_mask(X, predict, y, batch):
    x = X[0]
    x_ = predict[0]
    la = y[0]
    # 3 images with order: input, label, result from Unet
    img = torch.stack([x, la, x_], 0)
    save_image(img.cpu(), os.path.join('./result_img/', f"{batch}.png"))

    if batch == len(X) - 1:  # check the print only one time
        print("Image save successfully!")

    return 0


def plot_training_result(loss_values, iou_values, epochs):
    current_ep = 1

    while current_ep <= epochs:
        if current_ep == epochs:
            epochs_range = range(1, len(loss_values) + 1)
            plt.plot(epochs_range, loss_values, color='blue', label='Loss')
            plt.plot(epochs_range, iou_values, color='green', label='IoU')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Training Result')
            plt.legend()
            plt.xticks(epochs_range)  # Set the x-axis tick locations to the epochs_range
            plt.show()
        else:
            current_ep += 1


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

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    return bounding_boxes, labels, width, height, polygons


def polygon_to_mask(labels):
    labels = labels.tolist()
    current_label = None
    polygon_points = []
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # figsize =(3.2, 3.2) if image size = 320 * 320
    for point in labels:
        x, y, label = point

        if label != current_label:
            if polygon_points:
                polygon = Polygon(polygon_points, closed=True, alpha=0.5, facecolor=color)
                ax.add_patch(polygon)
            polygon_points = []
            current_label = label
            color = 'green' if label == 0 else 'red'

        polygon_points.append([x, y])

    if polygon_points:
        polygon = Polygon(polygon_points, closed=True, alpha=0.5, facecolor=color)
        ax.add_patch(polygon)
    ax.set_facecolor('black')
    ax.set_xlim(0, 640)  # Set x-axis limits
    ax.set_ylim(0, 480)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Label Mask')
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert Y-axis for correct visualization
    # plt.show()  # for show mask

    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_array = torch.from_numpy(image_array)
    plt.close(fig)  # Close the figure to free up resources

    return image_array


def func(batch):
    img, label = zip(*batch)
    img_list = []
    label_list = []
    for i, l in enumerate(label):
        img_ptr = torch.full((l.size(0), 1), i, dtype=torch.int32)
        l = torch.cat((img_ptr, l), dim=1)
        return [torch.stack(img, 0), torch.cat(label, 0)]


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
        # result = result[None, :]
        # print(result.size())  # check how many label of one images  e.g [239,3]
        lbl_size_tracker.append(result.size(dim=0))
        return img, result

    def __len__(self):
        return len(self.img_idx)


def pad_mask(size, mask):
    result_list = []
    # print(lbl_size_tracker)
    for lbl_track in lbl_size_tracker:
        temp = mask[:lbl_track]
        temp_poly = single_mask_trans(temp)
        result_list.append(temp_poly)

    combined_tensor = torch.cat(result_list, dim=0)

    return combined_tensor


def single_mask_trans(mask):
    mask = polygon_to_mask(mask)
    mask = np.transpose(mask, (2, 0, 1))
    mask = normalize(mask)
    mask = mask[None, :]
    return mask


def save_model(total_ep, loss, iou, model):
    folder = './model'
    best_loss = float('inf')
    best_iou = float('-inf')

    # file looping
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        # Check if there is file
        if os.path.isfile(file_path):
            # Extract the history loss and IoU of the 'model' folder
            history_loss = float(filename.split("_")[2][:-3])
            history_iou = float(filename.split("_")[3][:-3])
            # Check if the history loss < best loss and history IoU > best IoU
            if history_loss < best_loss and history_iou > best_iou:
                best_loss = history_loss
                best_iou = history_iou

    # Obtained the best result of train records, check if the current loss < best loss and current IoU > best IoU
    if loss < best_loss and iou > best_iou:
        # Specify the file path with proper formatting
        model_path = './model/model_{}_{}_{}.pt'.format(total_ep, loss, iou)
        # Save the model
        torch.save({'model': model.state_dict()}, model_path)
        print("model is saved !")


# Train Part
def train_loop(dataloader, model, loss_fn, optimizer, device, epochs, current_ep):
    size = len(dataloader.dataset)
    loss_values = []
    iou_values = []
    acc_values = []
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Data processing
        X = X.to(device)

        if X.size(dim=0) > 1:
            y = pad_mask(X.size(dim=0), y)
            lbl_size_tracker.clear()
        elif X.size(dim=0) == 1:
            y = single_mask_trans(y)
            lbl_size_tracker.clear()

        y = y.to(device)

        # Compute prediction and loss
        predict_x = model(X)
        # display_mask(X, predict_x, y, batch)  # for show output image and mask

        loss = loss_fn(predict_x, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Calculate Loss rate,F1 score and accuracy
        loss, current = loss.item(), (batch + 1) * len(X)
        true_mask = y.detach().cpu().numpy()
        prediction_x = predict_x.detach().cpu().numpy()
        true_mask = np.argmax(true_mask, axis=1)
        prediction_x = np.argmax(prediction_x, axis=1)

        f1 = f1_score(true_mask.reshape(-1), prediction_x.reshape(-1), average='macro')  # all became 1D array
        iou = jaccard_score(true_mask.reshape(-1), prediction_x.reshape(-1), average='macro')  # for segmentation task
        accuracy = accuracy_score(true_mask.reshape(-1), prediction_x.reshape(-1))  # for classification task
        loss_values.append(loss)
        iou_values.append(iou)
        acc_values.append(accuracy)

        print(
            f"loss: {loss:>6f}  F1 score: {f1:.4f}  "
            f"Accuracy: {accuracy * 100:.2f}%  "
            f"IoU: {iou * 100:.4f}%  "
            f"[{current:>5d}/{size:>5d}]")

        save_model(epochs, loss, iou, model)

        if (batch + 1) % len(dataloader) == 0:
            avg_loss = np.mean(loss)
            print(f"\nAverage Loss of Epoch {current_ep}: {avg_loss:.6f}")
            # plot_training_result(loss_values, iou_values, epochs)

    return loss_values,acc_values


import math


def create_lambda_scheduler(optimizer, warm_up_iter, T_max, lr_max, lr_min):
    def lambda_scheduler(cur_iter):
        if cur_iter < warm_up_iter:
            return cur_iter / warm_up_iter
            print("Warm up Learning rate processing.")
        else:
            cos_val = torch.cos(torch.tensor((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))
            return (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos_val)) / lr_max  # learning rate result

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_scheduler)
    return scheduler


def main():
    root_dir = "C:/Users/willi/PycharmProjects/modelling/Dataset/train/"
    train_data = Read_voc(root_path=root_dir)  # DataSet Preprocessing
    device = check_cpu()  # checking the training module is using cpu/gpu?
    # display_image_with_polygon(train_data)                                        # Display image and label.

    """display_image_with_boxes(img, res, label)"""
    """train_dataset, test_dataset = random_split(  # 70% for train set , 30% for test set
        dataset=train_data,
        lengths=[90, 60],
        generator=torch.Generator().manual_seed(0)
    )"""
    # Pytorch load Data
    batch_size = 6
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=func)
    # train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, collate_fn=func)
    # test_dataloader = DataLoader(test_dataset, batch_size=b_size, shuffle=True, collate_fn=func)

    # Display image and label.
    """ count = 5
    for index, data in enumerate(train_dataloader):
        features, labels = data
        print(f"Feature batch shape: {features.size()}")
        print(f"Labels batch shape: {labels.size()}")
        count = count + 1
    print(count) """
    # print(len(train_dataloader))         #check the integrity of dataset loader
    # print(len(test_dataloader))

    # Model Configuration
    classes = ['leaf', 'stem']

    model = unet.UNet(n_channels=3, n_classes=len(classes) + 1)
    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=0.0005)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)                      #logistic scheduler
    scheduler = create_lambda_scheduler(optimizer, warm_up_iter=2, T_max=20, lr_max=0.1, lr_min=1e-5)
    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # decrease LR after each 3 epochs
    cos_lr_list = []
    epochs = 50  # "time of training loop" value
    print(f" ***** Training Start ***** ")
    loss_total = []
    acc_total = []
    for t in range(epochs):
        start_time = time.time()
        print(f"Epoch {t + 1}\n———————————————————————————————Batch size: {batch_size}")
        current_loss,current_acc = train_loop(train_dataloader, model, loss_fn, optimizer, device, epochs, t + 1)
        print("The %d of epoch Learning rate：%f" % (t + 1, optimizer.param_groups[0]['lr']))
        cos_lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step()  # for adjust the learning rate because if not it will fix lr.
        # test_loop(test_dataloader, model, loss_fn, device, epochs)
        end_time = time.time()
        print("Epoch End ————Train time in this epoch: ———— {:.2f}".format((end_time - start_time)), 'seconds')
        print("———————————————————————————————————————————————")
        loss_total.extend(current_loss)
        acc_total.extend(current_acc)
    print(f" ***** Training End ***** ")

    import matplotlib.pyplot as plt

    # Plot the loss_total
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_total)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.title('Training Loss')

    # Plot the acc_total
    plt.figure()
    plt.plot(range(1, epochs + 1), acc_total)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')

    # Display both plots
    plt.show()



main()
