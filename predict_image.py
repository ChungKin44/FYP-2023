import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import unet as model

import datetime

# for the polygon
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Label_class = [[255, 255, 255], [0, 255, 0], [0, 0, 255]]  # others=white, leaf=green, stem=blue


class Predict:
    @staticmethod
    def predict_img(img_path):
        model_path = "./model/model_30_1.46429443359375_0.2634585257005816.pt"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the trained model
        net = model.UNet(n_channels=3, n_classes=3)
        state_dict = torch.load(model_path)

        # Check if the state_dict is nested and access it
        if 'model' in state_dict:
            state_dict = state_dict['model']

        net.load_state_dict(state_dict)
        net = net.to(device)

        # Load the image
        img = Image.open(img_path)

        # Apply the necessary transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        # Set the model to evaluation mode
        net.eval()

        # Forward pass
        with torch.no_grad():
            output = net(img)
            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

        # Convert probability to class
        mask = np.argmax(output, axis=1)        # convert to class mask

        # Define the colors for each class (others=white, leaf=green, stem=blue)
        colors = np.array(Label_class)

        # Convert the class mask to RGB
        mask_rgb = colors[mask[0]]

        # Convert the mask to a PIL Image and save it
        mask_img = Image.fromarray(mask_rgb.astype('uint8'))
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate the file name with the current time and date
        file_name = f"./prediction_image/mask_{current_time}.png"
        mask_img.save(file_name)
        print("Image saved! \n")

        return mask[0]

    @staticmethod
    def calculate_coverage(mask, class_index):
        # Calculate the coverage rate
        leaf_pixels = np.count_nonzero(mask == class_index)
        stem_pixels = np.count_nonzero(mask == class_index + 1)
        coverage_rate = leaf_pixels / (leaf_pixels + stem_pixels)

        # Find the contours and bounding boxes
        contours, _ = cv2.findContours((mask == class_index).astype('uint8'), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        return coverage_rate

    @staticmethod
    def result_bounding_box(mask, class_index):
        contours, _ = cv2.findContours((mask == class_index).astype('uint8'), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        return bounding_boxes

    @staticmethod
    def show_leaf_bb_result(img_path, bounding_boxes):
        # Load the original image
        original_img = Image.open(img_path)

        # Create a figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(original_img)
        ax.set_title("Leaf prediction")
        # Sort the bounding boxes by area in descending order
        sorted_boxes = sorted(bounding_boxes, key=lambda box: box[2] * box[3], reverse=True)

        # Create a Rectangle patch for each bounding box and add it to the Axes
        for i, box in enumerate(sorted_boxes[:5]):
            x, y, width, height = box
            if i == 0:  # The first (biggest) bounding box
                color = 'blue'
            else:
                color = 'red'
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Show the figure
        plt.show()


def main():
    img_path = "./demo_image/0814_1710.jpg"
    mask = Predict.predict_img(img_path)
    leaf = Label_class.index([0, 255, 0])
    coverage_rate = Predict.calculate_coverage(mask, leaf)
    print("Result  ———————————————————————————————————————\n")
    print(f'1. Coverage rate  : {coverage_rate * 100:.1f} %')
    print(f'2. Current height : ')
    print(f'3. Current area   : \n')
    print("———————————————————————————————————————————————")
    bounding_box = Predict.result_bounding_box(mask, leaf)
    #Predict.show_polygon(polygon)
    Predict.show_leaf_bb_result(img_path, bounding_box)


main()
