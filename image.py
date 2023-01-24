"""
Created By:
    Owen Doyle
    Ayleen Roque
"""



import json
import os
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from scipy.io import savemat
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Function is pretty much the same from this page:
# https://towardsdatascience.com/slicing-images-into-overlapping-patches-at-runtime-911fa38618d7
# Near the end, there is sample code for how to label the patches, we can implement after finishing up the slicing
# Use multithreading/multiprocessing to work on different data sets at once


"""
We can define a function to calculate how many slices we would need for a given image depending on its dimensions,
and return the coordinates for each slice as a bounding box in xyxy format
"""

def calculate_train_slice(image_height: int,  # height of the original image
                          image_width: int,  # width of the original image
                          slice_height: int,  # height of each slice
                          slice_width: int,  # width of each slice
                          overlap_height_ratio: float = 0,  # fractional overlap in height of each slice
                          overlap_width_ratio: float = 0,  # Fractional overlap in width of each slice
                          ) -> list[list[int]]:
    """Slices `image_pil` in crops.
        Corner values of each slice will be generated using the `slice_height`,
        `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.
        Args:
            image_height (int): Height of the original image.
            image_width (int): Width of the original image.
            slice_height (int): Height of each slice. Default 512.
            slice_width (int): Width of each slice. Default 512.
            overlap_height_ratio(float): Fractional overlap in height of each
                slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                overlap of 20 pixels). Default 0.
            overlap_width_ratio(float): Fractional overlap in width of each
                slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
                overlap of 20 pixels). Default 0.
        Returns:
            List[List[int]]: List of 4 corner coordinates for each N slices.
                [
                    [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                    ...
                    [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
                ]
        """
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height / 2:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height / 2 or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height / 2, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def calculate_valid_slice(image_height: int,  # height of the original image
                          image_width: int,  # width of the original image
                          slice_height: int,  # height of each slice
                          slice_width: int,  # width of each slice
                          overlap_height_ratio: float = 0,  # fractional overlap in height of each slice
                          overlap_width_ratio: float = 0,  # Fractional overlap in width of each slice
                          ) -> list[list[int]]:
    slice_bboxes = []
    y_max = y_min = image_height / 2
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height * .75:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height * .75 or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height * .75, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def calculate_test_slice(image_height: int,  # height of the original image
                         image_width: int,  # width of the original image
                         slice_height: int,  # height of each slice
                         slice_width: int,  # width of each slice
                         overlap_height_ratio: float = 0,  # fractional overlap in height of each slice
                         overlap_width_ratio: float = 0,  # Fractional overlap in width of each slice
                         ) -> list[list[int]]:
    slice_bboxes = []
    y_max = y_min = image_height * .75
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def get_rectangle_params_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_bboxes(
        plot_ax,
        bboxes,
        class_labels,
        get_rectangle_corners_fn=get_rectangle_params_from_pascal_bbox,
):
    for bbox, label in zip(bboxes, class_labels):
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left, width, height, linewidth=4, edgecolor="black", fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left, width, height, linewidth=2, edgecolor="white", fill=False,
        )
        rx, ry = rect_1.get_xy()

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)
        plot_ax.annotate(label, (rx + width, ry + height), color='white', fontsize=20)


def show_image(image, bboxes=None, class_labels=None, draw_bboxes_fn=draw_bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap=plt.get_cmap('gray'))

    if bboxes:
        draw_bboxes_fn(ax, bboxes, class_labels)

    plt.show()


def create_training_data(train_slices, img) -> np.array:
    train_slice_index = 0
    train_data = []

    for f in train_slices:
        xmin, ymin, xmax, ymax = train_slices[train_slice_index]
        # show_image(img[ymin:ymax, xmin:xmax])
        image_slice = img[ymin: ymax, xmin:xmax]
        train_data = np.asarray(image_slice)

        # train_data.append(np.asarray(image_slice))
        with open("train_data.txt", 'a') as array:
            pass
            array.write(f"Slice {train_slice_index}\n")
            array.write(' \n'.join(str(e) for e in train_data))
            array.write('\n')
            image = Image.fromarray(train_data)
            #image.show()
        train_slice_index += 1
    #print(train_data)
    return train_data
    # mdic = {data": data, "label": "training_set"}
    # savemat("matlab_training_set.mat", mdic)


def create_valid_data(valid_slices, img) -> np.array:
    valid_slice_index = 0
    valid_data = []
    with open("valid_data.txt", 'w'):
        pass
    for f in valid_slices:
        xmin, ymin, xmax, ymax = valid_slices[valid_slice_index]
        # show_image(img[ymin:ymax, xmin:xmax])
        image_slice = img[int(ymin): int(ymax), int(xmin):int(xmax)]
        valid_data = np.asarray(image_slice)
        # train_data.append(np.asarray(image_slice))
        with open("valid_data.txt", 'a') as array:
            pass
            array.write(f"Slice {valid_slice_index}\n")
            array.write(' \n'.join(str(e) for e in valid_data))
            array.write('\n')
            image = Image.fromarray(valid_data)
            # image.show()
        valid_slice_index += 1
    # mdic = {"data": data, "label": "training_set"}
    # savemat("matlab_training_set.mat", mdic)


def create_test_data(test_slices, img) -> np.array:
    test_slice_index = 0
    test_data = []
    with open("test_data.txt", 'w'):
        pass
    for f in test_slices:
        xmin, ymin, xmax, ymax = test_slices[test_slice_index]
        # show_image(img[ymin:ymax, xmin:xmax])
        image_slice = img[int(ymin): int(ymax), int(xmin):int(xmax)]
        test_data = np.asarray(image_slice)
        # train_data.append(np.asarray(image_slice))
        with open("test_data.txt", 'a') as array:
            pass
            array.write(f"Slice {test_slice_index}\n")
            array.write(' \n'.join(str(e) for e in test_data))
            array.write('\n')
            image = Image.fromarray(test_data)
            # image.show()
        test_slice_index += 1
    # mdic = {"data": data, "label": "training_set"}
    # savemat("matlab_training_set.mat", mdic)

def one_hot():
    images = []
    for files in os.listdir("./grey_images"):
        images.append(files)

    label = LabelEncoder()
    int_data = label.fit_transform(images)
    int_data = int_data.reshape(len(int_data), 1)

    onehot_data = OneHotEncoder(sparse_output=False)
    onehot_data = onehot_data.fit_transform(int_data)
    return onehot_data
    #print(images)


def convert_to_grayscale(directory):
    for image in os.listdir(directory):
        # Creating an og_image object
        og_image = Image.open(directory + '/' + image)

        # Applying grayscale method
        gray_image = ImageOps.grayscale(og_image)

        # Save image to seperate directory
        gray_image.save('./grey_images/' + image)


########################################################################################################################

# Directories
image_directory = "/home/owen/PycharmProjects/labProject/converted_images/"
train_directory = "/home/owen/PycharmProjects/labProject/training_set"
valid_directory = "/home/owen/PycharmProjects/labProject/valid_set"
test_directory = "/home/owen/PycharmProjects/labProject/test_set"

""" load the data (numpy array) -- starting with one image """
# load the image and convert into numpy array
#gimg = Image.open('./converted_images/merry_flor0019.jpg').convert('L')
#gimg.save('./converted_images_grey/greyscale.jpg')
#img = plt.imread('./converted_images_grey/greyscale.jpg')
#print(img.shape)
# img = plt.imread('./converted_images/merry_flor0019.jpg')
# type(img)
# print(img.shape)
direc = "./tiff_images"
convert_to_grayscale(direc)

with open("train_data.txt", 'w'):
    pass

nparray = []
with open("imagearray.txt", 'w') as array:
    pass
for image in os.listdir('./grey_images'):
    img = (plt.imread('./grey_images/' + image))
    nparray.append(img)
    with open("imagearray.txt", 'a') as array:
        for data in img:
            array.write(
                f"{data}"
            )
        array.write("\n\n")

image_data_train = []

for i in range(len(nparray)):
    img = nparray[i]

    # set parameters
    image_height = img.shape[0]
    image_width = img.shape[1]
    slice_height: int = 64
    slice_width: int = 64
    overlap_height_ratio: float = 0  # currently set overlap to none for simplicity
    overlap_width_ratio: float = 0

    """ apply sliding window algorithm """

    train_slices = calculate_train_slice(image_height, image_width, slice_height, slice_width, overlap_height_ratio,
                                         overlap_width_ratio)

    valid_slices = calculate_valid_slice(image_height, image_width, slice_height, slice_width, overlap_height_ratio,
                                         overlap_width_ratio)

    test_slices = calculate_test_slice(image_height, image_width, slice_height, slice_width, overlap_height_ratio,
                                       overlap_width_ratio)

    show_image(img)


    image_data_train.append(create_training_data(train_slices, img))





one_hot_images = []
one_hot_images.append(one_hot())
print(one_hot_images)

combined_data = tuple(zip(one_hot_images, image_data_train))

print(combined_data)



