from math import sqrt

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

import convexhull_grScan
import Unet as model

import datetime

# for the polygon
import cv2
import numpy as np
import matplotlib.pyplot as plt

# shape polygon
import convexhull_grScan as convexhull_gr

Label_class = [[255, 255, 255], [0, 255, 0], [0, 0, 255]]  # others=white, leaf=green, stem=blue
leaf = Label_class.index([0, 255, 0])
stem = Label_class.index([0, 0, 255])


class Predict:
    def __init__(self):
        self.model_path = "./model/well_performance/model_10_0.8625010850694445_0.5217529723043653.pt"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = model.UNet(n_channels=3, n_classes=3).to(self.device)
        self.colors = np.array(Label_class)

    def predict_img(self, img_path):
        # Load the trained model
        state_dict = torch.load(self.model_path)

        # Check if the state_dict is nested and access it
        if 'model' in state_dict:
            state_dict = state_dict['model']

        self.net.load_state_dict(state_dict)

        # Load the image
        img = Image.open(img_path)

        # Apply the necessary transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = img.convert("RGB")   # make sure if it is RGB image
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        # Set the model to evaluation mode
        self.net.eval()

        # Forward pass
        with torch.no_grad():
            output = self.net(img)
            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()

        # Convert probability to class
        mask = np.argmax(output, axis=1)  # convert to class mask

        # Convert the class mask to RGB
        mask_rgb = self.colors[mask[0]]

        # Convert the mask to a PIL Image and save it
        mask_img = Image.fromarray(mask_rgb.astype('uint8'))
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate the file name with the current time and date
        file_name = f"./prediction_image/mask_{current_time}.png"
        mask_img.save(file_name)
        print("Image saved! \n")

        return mask[0]

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


class Contour(Predict):

    def analyse_polygon(self, img, mask, leaf, bound_box):
        contours, _ = cv2.findContours((mask == leaf).astype('uint8'), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

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

        return leaf_set

    def calculate_radius(self, leaf_set):
        max_radius = 0
        for leaf in leaf_set:
            x, y = leaf[0][0], leaf[0][1]
            radius = np.sqrt(x ** 2 + y ** 2)
            if radius > max_radius:
                max_radius = radius
        return max_radius

    def show_contour(self, img, mask, leaf, bound_box):
        leaf_set = self.analyse_polygon(img, mask, leaf, bound_box)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        radius = self.calculate_radius(leaf_set)
        # print("Radius of the leaf set:", radius)

        # Draw and line up all the pixels of the contour
        # convex_hull = cv2.convexHull(np.array(leaf_set), returnPoints=True)
        # Example usage:
        # Approximate the convex hull with a more precise polygon
        # epsilon = 0.01 * cv2.arcLength(convex_hull, True)
        # polygon = cv2.approxPolyDP(convex_hull, epsilon, True)


        # Convex Hull using Graham Scan for polygon
        convex_hull_leaf = np.array(convexhull_grScan.ConvexHull().compute_convex_hull(leaf_set),
                                    dtype=np.int32).reshape((-1, 1, 2))

        # Draw convex hull
        cv2.polylines(img, [convex_hull_leaf], True, (0, 255, 0), 2)
        # Show the image
        cv2.imshow("window", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return convex_hull_leaf


class Analyse_attribute(Predict):

    def calculate_coverage(self, mask, class_index):
        # Calculate the coverage rate
        leaf_pixels = np.count_nonzero(mask == class_index)
        stem_pixels = np.count_nonzero(mask == class_index + 1)
        coverage_rate = leaf_pixels / (leaf_pixels + stem_pixels)
        return coverage_rate

    def polygon_area(self, leaf_polygon):  # its underline because the method was only call in the class
        n = len(leaf_polygon)
        area = 0
        for i in range(n):
            x1, y1 = leaf_polygon[i][0][0], leaf_polygon[i][0][1]
            x2, y2 = leaf_polygon[(i + 1) % n][0][0], leaf_polygon[(i + 1) % n][0][1]
            area += (x1 * y2 - x2 * y1)

        return abs(area) / 2.0

    def real_area(self, area):
        # camera spec.
        s_width = 3.59  # in mm unit
        s_height = 2.684

        # image spec.
        im_width = 640  # in pixel unit
        im_height = 480

        # environment parameter
        actual_distance = 20  # cm unit
        focal_len = 0.36  # cm unit

        pixel_width = (s_width / im_width)
        pixel_height = (s_height / im_height)
        sensor_area = (sqrt(area) * pixel_height) * (sqrt(area) * pixel_width)  # mm^2 unit
        actual_area = sensor_area * actual_distance / focal_len  # cm^2 unit
        return actual_area

    def print_analyse(self, mask, label_0, leaf_polygon):
        coverage_rate = self.calculate_coverage(mask, label_0)  # label_0 = leaf
        area = self.polygon_area(leaf_polygon)
        actual_area = self.real_area(area)

        print("Result  ———————————————————————————————————————")
        print(f'1. Coverage rate  : {coverage_rate * 100:.1f} %')
        print(f'2. Current height : ')
        print(f'3. Current actual area   : {actual_area:.2f} cm²')
        # print(f'3. Current area ratio  : {area_ratio:.1f} of the photo')
        print("———————————————————————————————————————————————")


def main():
    img_path = "./demo_image/single_image/vertical.jpg"
    img = cv2.imread(img_path)

    predictor = Predict()
    analyser = Analyse_attribute()
    contour = Contour()

    mask = predictor.predict_img(img_path)

    # Object detection class
    bounding_box = predictor.result_bounding_box(img, mask, leaf)
    bb = predictor.show_leaf_bb_result(img_path, bounding_box)

    # Contour class
    ol_bbox = predictor.overlap_bb(bb)
    leaf_polygon = contour.show_contour(img, mask, leaf, ol_bbox)

    # Final : analyse class
    analyser.print_analyse(mask, leaf, leaf_polygon)


main()
