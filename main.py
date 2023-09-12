# Preprocess
import os

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


def display_image_with_boxes(image, boxes, lbls):
    img_with_boxes = image.clone().permute(1, 2, 0).numpy()
    plt.imshow(img_with_boxes)
    ax = plt.gca()
    for bbox in boxes:
        xmin, ymin, xmax, ymax = bbox
        color = 'blue' if lbls == 0 else 'red'  # blus as leaf , red as stem
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
    plt.show()


voc_label = {'leaf,stem,soil'}
dict_labels = dict(zip(voc_label, range(len(voc_label))))


def func(batch):  # https://blog.csdn.net/weixin_44326452/article/details/123015556
    img, label = zip(*batch)
    for i, l in enumerate(label):
        l[:, 0] = i
    return torch.stack(img, 0), torch.cat(label, 0)


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
        print(self.img_idx)
        img = Image.open(self.img_idx[item])
        img = transforms.ToTensor()(img)
        normalize(img)

        res = []  # Store annotation information, i.e., coordinates of the bounding box's top left and bottom right
        # points and the target's class label
        result = []
        la = []

        if os.path.exists(self.root_path):
            bboxes, labels, width, height = parse_xml(self.anno_idx[item])

            lbls = convert_label(labels)  # Convert label to number using convert_label() function
            res.append(bboxes)
            res.append(lbls)
            for bbox, label in zip(*res):
                result.append(bbox + [label])
            la.append(lbls)


        else:
            raise Exception('Path does not Exist!')

        result = torch.from_numpy(np.array(result))
        """print(result.size())    #check how many label of one images"""

        return img, result

    def __len__(self):
        return len(self.img_idx)


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, scheduler, devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.long(), loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        scheduler.step()

        print(
            f"epoch {epoch + 1} --- loss {metric[0] / metric[2]:.3f} ---  train acc {metric[1] / metric[3]:.3f} --- test acc {test_acc:.3f} --- cost time {timer.sum()}")

        # ---------保存训练数据---------------
        """ df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch)
        time_list.append(timer.sum())

        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df['time'] = time_list
        df.to_excel("C:/Users/willi/OneDrive/桌面/Final Year Project/Unet++_camvid1.xlsx")
        # ----------------保存模型-------------------
        if np.mod(epoch + 1, 5) == 0:
            torch.save(model.state_dict(), f'C:/Users/willi/OneDrive/桌面/Final Year Project/Unet++_{epoch + 1}.pth') """


def main():
    root_dir = "C:/Users/willi/OneDrive/桌面/Dataset/train/"
    train_data = Read_voc(root_path=root_dir)  # DataSet Preprocessing
    device = check_cpu()             #checking the training module is using cpu/gpu?
    """print(type(res))
    print(img.size())
    print(train_data.__len__())"""
    train_dataset, test_dataset = random_split(
        dataset=train_data,
        lengths=[100, 50],
        generator=torch.Generator().manual_seed(0)
    )
    # display_image_with_boxes(img, res, lbls)
    # Display image and label.
    size = 2
    train_dataloader = DataLoader(train_dataset, batch_size=size, shuffle=True, collate_fn=func)
    test_dataloader = DataLoader(test_dataset, batch_size=size, shuffle=True, collate_fn=func)
    count = 0
    """" for index, data in enumerate(train_dataloader):
        feature, labels = data
        print(feature.size())
        count = count + 1
    print(count) """
    # print(len(train_dataloader))         #check the integrity of dataset loader
    # print(len(test_dataloader))

    # Model Configuration
    model = unet.UNet(n_channels=3, n_classes=1 + 3, bilinear=True).to(device)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=1e-5, weight_decay=1e-8, momentum=0.999)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    epochs_num = 5
    # Train start
    train_ch13(model, train_dataloader, test_dataloader, criterion, optimizer, epochs_num, scheduler)


main()
