import csv
import pandas as pd
import torchvision.transforms.functional as F
import numpy as np
import os
import shutil
from PIL import Image

# Adjustable parameters
saturation_factor = 1.8  # Saturation adjustment factor
brightness_factor = 1.2  # Brightness adjustment factor
rotation_degree = 15  # Rotation degree
noise = 0.02  # 2%

# For augmentation
# set the input folder path
img_folder = '../Dataset/final/train'

# set the output folder path
augmented_folder = '../Dataset/final/augmented'

# Annotate Progress
# set the folder path
folder_path = augmented_folder
# set the output path
output_file = os.path.join(augmented_folder, 'file_names.csv')

if not os.path.exists(augmented_folder):
    os.makedirs(augmented_folder)


def original_copy(source, destination):
    for filename in os.listdir(source):

        filename_no_ext = os.path.splitext(filename)[0]

        if filename.endswith('.jpg'):
            src = os.path.join(source, filename)
            dest = os.path.join(destination, "ol_" + filename)
            shutil.copy(src, dest)

        elif filename.endswith('.xml'):
            src = os.path.join(source, filename)
            dest = os.path.join(destination, "ol_" + filename)
            shutil.copy(src, dest)


def modify_saturation(img):
    modified1 = F.adjust_saturation(img, saturation_factor)
    modified2 = F.adjust_saturation(img, 1 / saturation_factor)
    return [("sa1_", modified1), ("sa2_", modified2)]


def augment_images_sat(source, dest):
    for filename in os.listdir(source):
        if filename.endswith('.jpg'):
            filename_no_ext = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(source, filename))
            modifications = modify_saturation(img)

            for prefix, modified in modifications:
                dest_image = os.path.join(dest, prefix + filename)
                modified.save(dest_image)

                xml_file = filename_no_ext + ".xml"
                src_xml = os.path.join(source, xml_file)
                dest_xml = os.path.join(dest, prefix + xml_file)
                shutil.copy(src_xml, dest_xml)


def modify_brightness(img):
    modified1 = F.adjust_brightness(img, brightness_factor)  # +60%
    modified2 = F.adjust_brightness(img, 1 / brightness_factor)  # -60%
    return [("bs1_", modified1), ("bs2_", modified2)]


def augment_images_bs(source, dest):
    for filename in os.listdir(source):
        if filename.endswith('.jpg'):
            filename_no_ext = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(source, filename))
            modifications = modify_brightness(img)

            for prefix, modified in modifications:
                dest_image = os.path.join(dest, prefix + filename)
                modified.save(dest_image)

                xml_file = filename_no_ext + ".xml"
                src_xml = os.path.join(source, xml_file)
                dest_xml = os.path.join(dest, prefix + xml_file)
                shutil.copy(src_xml, dest_xml)


def flip_image(img):
    flipped_horizontal = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_vertical = img.transpose(Image.FLIP_TOP_BOTTOM)
    return [("hor_", flipped_horizontal), ("ver_", flipped_vertical)]


def augment_images_flip(source, dest):
    for filename in os.listdir(source):
        if filename.endswith('.jpg'):
            filename_no_ext = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(source, filename))
            flipped_images = flip_image(img)

            for prefix, flipped_img in flipped_images:
                dest_image = os.path.join(dest, prefix + filename)
                flipped_img.save(dest_image)

                xml_file = filename_no_ext + ".xml"
                src_xml = os.path.join(source, xml_file)
                dest_xml = os.path.join(dest, prefix + xml_file)
                shutil.copy(src_xml, dest_xml)


def add_noise(image, noise_percentage):
    np.random.seed(42)  # Set a seed for reproducibility
    img_array = np.array(image)  # Convert the image to a NumPy array

    # Calculate the number of pixels to modify based on the noise percentage
    num_pixels = int(noise_percentage * img_array.size)

    # Randomly select pixels to add noise to
    indices = np.random.choice(img_array.size, num_pixels, replace=False)

    # Add noise to the selected pixels
    img_array.flat[indices] = np.random.randint(0, 256, size=num_pixels)

    # Convert the modified NumPy array back to an image
    noisy_image = Image.fromarray(img_array)

    return [("noise_", noisy_image)]


def augment_images_noise(source, dest):
    for filename in os.listdir(source):
        if filename.endswith('.jpg'):
            filename_no_ext = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(source, filename))
            noise_images = add_noise(img, noise)

            for prefix, flipped_img in noise_images:
                dest_image = os.path.join(dest, prefix + filename)
                flipped_img.save(dest_image)

                xml_file = filename_no_ext + ".xml"
                src_xml = os.path.join(source, xml_file)
                dest_xml = os.path.join(dest, prefix + xml_file)
                shutil.copy(src_xml, dest_xml)


def rotate_image(img, angle):
    rotated_img = img.rotate(angle)
    return rotated_img


def rotate_image(img, angle):
    rotated_img = img.rotate(angle)
    return rotated_img


def augment_images_rotate(source, dest):
    for filename in os.listdir(source):
        if filename.endswith('.jpg'):
            filename_no_ext = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(source, filename))

            pos_rotated_img = rotate_image(img, rotation_degree)  # Rotate +10 degrees
            neg_rotated_img = rotate_image(img, -rotation_degree)  # Rotate -10 degrees

            pos_rotated_filename = "rot_pos_10_" + filename
            pos_rotated_file_path = os.path.join(dest, pos_rotated_filename)
            pos_rotated_img.save(pos_rotated_file_path)

            neg_rotated_filename = "rot_neg_10_" + filename
            neg_rotated_file_path = os.path.join(dest, neg_rotated_filename)
            neg_rotated_img.save(neg_rotated_file_path)

            xml_file = filename_no_ext + ".xml"
            src_xml = os.path.join(source, xml_file)
            pos_dest_xml = os.path.join(dest, "rot_pos_10_" + xml_file)
            neg_dest_xml = os.path.join(dest, "rot_neg_10_" + xml_file)

            shutil.copy(src_xml, pos_dest_xml)
            shutil.copy(src_xml, neg_dest_xml)


def write_folder_names_to_csv(folder_path, output_file):
    # Get the list of file names in the folder
    file_names = os.listdir(folder_path)
    num_files = len(file_names)

    # Create or overwrite the output CSV file
    with open(output_file, 'w', newline='') as f_csv:
        # Create a CSV writer
        writer_csv = csv.writer(f_csv)

        # Create or overwrite the output text file
        output_text_file = os.path.splitext(output_file)[0] + ".txt"
        with open(output_text_file, 'w') as f_txt:
            # Write each file name as a row in the CSV file and text file with a counter
            for i, file_name in enumerate(file_names, start=1):
                if file_name.endswith('.jpg'):
                    writer_csv.writerow([file_name, i])
                    f_txt.write(file_name + '\n')

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(output_file)

    # Add a column for the total number of rows
    df.insert(2, 'Total Rows', num_files)

    # Save the modified DataFrame back to the CSV file
    df.to_csv(output_file, index=False)

    print(f"File names written to {output_file} and {output_text_file} successfully.")


def main():
    print("Augmentation Processing...")
    original_copy(img_folder, augmented_folder)
    augment_images_sat(img_folder, augmented_folder)
    augment_images_bs(img_folder, augmented_folder)
    augment_images_flip(img_folder, augmented_folder)
    augment_images_noise(img_folder, augmented_folder)
    #augment_images_rotate(img_folder, augmented_folder)
    print("Augmentation Done.")

    write_folder_names_to_csv(folder_path, output_file)


main()
