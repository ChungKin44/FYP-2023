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

# shape polygon
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

Label_class = [[255, 255, 255], [0, 255, 0], [0, 0, 255]]  # others=white, leaf=green, stem=blue


class Predict:
    @staticmethod
    def predict_img(img_path):
        model_path = "./model/model_30_1.452594518661499_0.29932511012332114.pt"
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
        mask = np.argmax(output, axis=1)  # convert to class mask

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

        return coverage_rate

    @staticmethod
    def result_bounding_box(img, mask, class_index):
        contours, hierarchy = cv2.findContours((mask == class_index).astype('uint8'), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        return bounding_boxes

    @staticmethod
    def show_leaf_bb_result(img_path, bounding_boxes):
        # Load the original image
        original_img = cv2.imread(img_path)

        # Create copies of the image for drawing bounding boxes
        img_with_boxes = original_img.copy()

        # Sort the bounding boxes by area in descending order
        sorted_boxes = sorted(bounding_boxes, key=lambda box: box[2] * box[3], reverse=True)

        # Draw the bounding boxes on the image
        for i, box in enumerate(sorted_boxes[:5]):
            x, y, width, height = box
            if i == 0:  # The first (biggest) bounding box
                color = (255, 0, 0)  # Blue color in BGR format
            else:
                color = (0, 0, 255)  # Red color in BGR format
            cv2.rectangle(img_with_boxes, (x, y), (x + width, y + height), color, 1)

        # Display the image with bounding boxes
        cv2.imshow("Leaf prediction", img_with_boxes)

        return sorted_boxes[1:5]

    @staticmethod
    def overlap_polygon(bound_box, merged_polygon):
        # find the overlapping labeled
        merged_polygon = [pair for pair in merged_polygon if
                          any(pair not in other_pair for other_pair in merged_polygon)]

        min_x, min_y, width, height = bound_box
        max_x = min_x + width
        max_y = min_y + height

        filtered_data = []
        for data_pair in merged_polygon:
            x = data_pair[0][0]
            y = data_pair[0][1]
            if min_x <= x <= max_x and min_y <= y <= max_y:
                filtered_data.append(data_pair)

        return filtered_data

    @staticmethod
    def show_polygon(img, mask, leaf, bound_box):
        contours, _ = cv2.findContours((mask == leaf).astype('uint8'), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        print(bound_box)
        leafs_polygons = []
        for contour in contours:
            # Perform optional filtering based on contour area or other criteria if needed
            # Approximate the contour with a polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            # Store the polygon for further processing if needed
            leafs_polygons.append(polygon)

        # Sort the polygons by area
        leafs_polygons.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        print(len(leafs_polygons))
        # Plot the n polygons on the image
        filter_polygons = []
        merged_polygon = []
        for polygon in leafs_polygons:
            if len(polygon) >= 4:
                filter_polygons.append(polygon)
            if len(polygon) == 1:
                for coord in polygon:
                    merged_polygon.append(coord)

        leaf_set = Predict.overlap_polygon(bound_box, merged_polygon)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for leaf in leaf_set:
            leaf_points = np.array(leaf, dtype=np.int32)
            cv2.polylines(img, [leaf_points], True, (255, 0, 0), 3)

        cv2.imshow("window", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def convert_white_blue_to_black(mask):
        for i in range(mask.shape[0]):
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] == 0 or mask[i, j] == 2:
                        mask[i, j] = 0  # Set the pixel value to black (0)
                    else:
                        mask[i, j] = 255  # Set the pixel value to white (255, or any other value for white)

                # Convert white pixels to green
            mask[mask > 0] = 65280  # Set the pixel value to the green color representation (65280)

            plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.show()

            return mask

    @staticmethod
    def overlap_bb(bboxes):
        # Extract the coordinates from the bounding boxes
        x_coord_s = [box[0] for box in bboxes]
        y_coord_s = [box[1] for box in bboxes]
        widths = [box[2] for box in bboxes]
        heights = [box[3] for box in bboxes]

        # Find the minimum and maximum coordinates
        min_x = min(x_coord_s)
        min_y = min(y_coord_s)
        max_x = max([x + w for x, w in zip(x_coord_s, widths)])
        max_y = max([y + h for y, h in zip(y_coord_s, heights)])

        # Calculate the width and height of the combined bounding box
        width = max_x - min_x
        height = max_y - min_y

        # Create the combined bounding box
        ol_bbox = (min_x, min_y, width, height)

        return ol_bbox


from PIL import Image


def main():
    img_path = "./demo_image/WhatsApp Image 2023-10-11 at 12.23.16_131b91f1.jpg"
    mask = Predict.predict_img(img_path)
    leaf = Label_class.index([0, 255, 0])
    coverage_rate = Predict.calculate_coverage(mask, leaf)
    print("Result  ———————————————————————————————————————\n")
    print(f'1. Coverage rate  : {coverage_rate * 100:.1f} %')
    print(f'2. Current height : ')
    print(f'3. Current area   : \n')
    print("———————————————————————————————————————————————")
    img = cv2.imread(img_path)
    bounding_box = Predict.result_bounding_box(img, mask, leaf)
    bb = Predict.show_leaf_bb_result(img_path, bounding_box)
    ol_bbox = Predict.overlap_bb(bb)
    Predict.show_polygon(img, mask, leaf, ol_bbox)
    # Call the function with the path to your image
    # Predict.convert_white_blue_to_black(mask)


main()
