import imageio.v2 as imageio
import numpy as np
import cv2
import kivy
from kivy.app import App
from kivy.uix.label import Label

def color_filter():
    image = imageio.imread('img.jpg')

    # Define the four colors
    colors = [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0] # Yellow
    ]

    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image)

    # Loop through each pixel of the original image
    #column
    for i in range(image.shape[0]):
        #row
        for j in range(image.shape[1]):
            pixel = image[i, j]

            # Calculate the Euclidean distance between the pixel and each color
            distances = [np.linalg.norm(pixel - color) for color in colors]

            # Get the index of the color with the minimum distance
            min_distance_index = np.argmin(distances)

            # Assign the corresponding color to the pixel in the filtered image
            filtered_image[i, j] = colors[min_distance_index]

    # Check if the width is larger than the height
    if image.shape[0] > image.shape[1]:
        # Rotate the image by -90 degrees
        filtered_image = np.rot90(filtered_image)

    # Save the filtered image
    filtered_image_path = 'filtered_image.jpg'
    imageio.imsave(filtered_image_path, filtered_image)

    return filtered_image_path

color_filter()

