# Preprocess
import os

from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR, StepLR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Basic Library
import numpy as np
import json
# 1. Preprocessing including load data
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib
from ash_Preprocess import Read_voc
import report

matplotlib.use('AGG')

# 2. training and model
import time
# import unet  # from unet.py
import Unet3plus
import torch.nn as nn
import indicators
from statistics import mean

torch.cuda.empty_cache()
lbl_size_tracker = []
# 2.1 training parameter
from sklearn.metrics import accuracy_score, jaccard_score


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


def save_model(total_ep, accuracy, iou, model, batch):
    model_folder = './Torch_Unet3plus/model'
    best_acc = float('-inf')
    best_iou = float('-inf')
    # file looping
    for filename in os.listdir(model_folder):
        file_path = os.path.join(model_folder, filename)
        # Check if there is a file
        if os.path.isfile(file_path):
            # Extract the history loss and IoU of the 'model' folder
            history_acc = float(filename.split("_")[3])
            history_iou = float(filename.split("_")[4])

            # Check if the history loss < best loss and history IoU > best IoU
            if history_acc > accuracy and history_iou > best_iou:
                best_acc = history_acc
                best_iou = history_iou

    # Obtained the best result of train records, check if the current acc > best acc and current IoU > best IoU
    if accuracy > best_acc and iou > best_iou:
        # Format the accuracy and IoU values with two decimal points
        accuracy_formatted = "{:.4f}".format(accuracy)
        iou_formatted = "{:.4f}".format(iou)
        # Specify the file path with proper formatting
        model_path = os.path.join(model_folder,
                                  'model_{}_{}_{}_{}_.pt'.format(total_ep, batch, accuracy_formatted, iou_formatted))
        # Save the model
        torch.save({'model': model.state_dict()}, model_path)
        print("Trained model is saved!")


def train_loop(dataloader, model, loss_fn, optimizer, device, epochs, current_ep):
    size = len(dataloader.dataset)
    loss_values = []
    iou_values = []
    acc_values = []
    dice_values = []

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # print("Image size : ", X.size())
        # print("Label size : ", y.size())

        # # Data processing  - to GPU
        X = X.to(device)
        y = y.float()

        y = y.to(device)

        predict_x = model(X).to(device)

        loss = loss_fn(predict_x, y).to(device)

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
        # Note : the label and mask is shape from 2D to 1D

        iou, dice_value = indicators.compute(true_mask.reshape(-1), prediction_x.reshape(-1))
        accuracy = accuracy_score(true_mask.reshape(-1), prediction_x.reshape(-1))  # for classification task
        loss_values.append(loss)
        iou_values.append(iou)
        acc_values.append(accuracy)
        dice_values.append(dice_value)

        print(
            f"Loss: {loss:>6f}  F1 score/Dice Coefficient: {dice_value * 100:.2f}%  "
            f"Accuracy: {accuracy * 100:.2f}%  "
            f"IoU: {iou * 100:.2f}%  "
            f" [{current:>5d}/{size:>5d}]"
        )

        save_model(epochs, accuracy, iou, model, batch)

        if (batch + 1) % len(dataloader) == 0:
            avg_loss = np.mean(loss)
            print(f"\nAverage Loss of Epoch {current_ep}: {avg_loss:.6f}")

    mean_loss_values = mean(loss_values)
    mean_iou_values = mean(iou_values)
    mean_acc_values = mean(acc_values)
    mean_dice_values = mean(dice_values)
    return mean_loss_values, mean_acc_values, mean_iou_values, mean_dice_values


import math


def create_lambda_scheduler(optimizer, warm_up_epochs, total_epochs, lr_max, lr_min):
    def lambda_scheduler(cur_iter):
        if cur_iter < warm_up_epochs:
            print("Warm up Learning rate processing.")
            return cur_iter / warm_up_epochs
        else:
            cos_val = torch.cos(torch.tensor((cur_iter - warm_up_epochs) / (total_epochs - warm_up_epochs) * math.pi))
            return (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos_val)) / lr_max  # learning rate result

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_scheduler)
    return scheduler


def main():
    # Select the train set
    # root_dir = "./Dataset/augmented/"
    root_dir = r'./Dataset/Ash/augmented/'
    train_data = Read_voc(root_path=root_dir)  # DataSet Preprocessing
    device = check_cpu()  # checking the training module is using cpu/gpu?
    print(f"Total images read: {len(train_data)}")

    # Pytorch load Data
    batch_size = 16
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=b_size, shuffle=True, collate_fn=func)

    # Model Configuration
    epochs = 200  # "time of training loop" value
    # model = Unet3plus.UNet3Plus_DeepSup_CGM()
    model = Unet3plus.UNet3Plus(num_classes=3)
    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.09)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)                      #logistic scheduler
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # scheduler = create_lambda_scheduler(optimizer, warm_up_epochs=20, total_epochs=epochs, lr_max=0.1, lr_min=1e-5)
    # cosine scheduler : warm up times , training times max

    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)  # decrease LR after each 3 epochs
    cos_lr_list = []

    print(f" ***** Training Start ***** ")

    loss_total = []
    acc_total = []
    dice_total = []
    iou_total = []
    for t in range(epochs):
        start_time = time.time()
        print(f"Epoch {t + 1}\n———————————————————————————————Batch size: {batch_size}")
        model.train()
        current_loss, current_acc, current_iou, current_dice = train_loop(train_dataloader, model, loss_fn, optimizer,
                                                                          device, epochs, t + 1)
        print("The %d of epoch Learning rate：%f" % (t + 1, optimizer.param_groups[0]['lr']))
        cos_lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print("Epoch End ———— Train time in this epoch: ———— {:d} minutes {:.2f} seconds".format(minutes, seconds))
        print("———————————————————————————————————————————————")

        loss_total.append(current_loss)
        acc_total.append(current_acc)
        dice_total.append(current_acc)
        iou_total.append(current_acc)
        torch.save(model.state_dict(), "./Torch_Unet3plus/epochs/Unet3plus_model_{}.pt".format(t + 1))

    print(f" ***** Training End ***** ")
    report.plot_loss_and_accuracy(loss_total, acc_total, iou_total, dice_total)

    total_report = {
        'loss_total': loss_total,
        'acc_total': acc_total,
        'dice_total': dice_total,
        'iou_total': iou_total
    }
    print(cos_lr_list)
    # Convert the dictionary to a JSON string
    json_str = json.dumps(total_report)

    # Specify the path and filename for the text file
    file_path = "total_report_warmupcosine.txt"

    # Write the JSON string to the file
    with open(file_path, 'w') as file:
        file.write(json_str)


main()
