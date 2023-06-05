from PIL import Image
from random import random, uniform
from numpy import array, matmul, float64
from math import sin, tan

def convert_grayscale_to_rgb(img: Image.Image):
    return Image.merge('RGB', (img, img, img))

def random_horizontal_flip(img: Image.Image, prob_flip: float = 0.5):
    if random() > prob_flip:
        return img

    width, height = img.size
    data = img.load()
    flipped_img = Image.new("L", (width, height))

    for h in range(height):
        for w in range(int(width / 2.0)):
            flipped_img.putpixel((w, h), data[width - 1 - w, h])
            flipped_img.putpixel((width - 1 - w, h), data[w, h])

    return flipped_img

def within_bounds(w: int, h: int, width: int, height: int):
    return w >= 0 and w < width and h >= 0 and h < height

def rotate(coords, three_shear_rotation):
    rotated_coords = coords
    for shear in three_shear_rotation:
        rotated_coords = matmul(shear, rotated_coords)
        # intermediate rounding prevents gaps in rotated image
        rotated_coords[0][0] = round(rotated_coords[0][0])
        rotated_coords[1][0] = round(rotated_coords[1][0])
    return rotated_coords

def random_rotation(img: Image.Image, lower_theta_bound: float = -0.349066,
                    upper_theta_bound: float = 0.349066):
    width, height = img.size
    data = img.load()
    rotated_img = Image.new("L", (width, height))

    theta = uniform(lower_theta_bound, upper_theta_bound)

    outer_shear = array([[1.0, -tan(theta / 2.0)], [0.0, 1.0]], float64)
    middle_shear = array([[1.0, 0.0], [sin(theta), 1.0]], float64)
    three_shear_rotation = [outer_shear, middle_shear, outer_shear]

    center_coords = array([[width / 2], [height / 2]])
    rotated_center_coords = rotate(center_coords, three_shear_rotation)
    x_shift = rotated_center_coords[0][0] - center_coords[0][0]
    y_shift = rotated_center_coords[1][0] - center_coords[1][0]

    for h in range(height):
        for w in range(width):
            coords = [[w], [h]]
            rotated_coords = rotate(coords, three_shear_rotation)
            rotated_x_centered = int(rotated_coords[0, 0] - x_shift)
            rotated_y_centered = int(rotated_coords[1, 0] - y_shift)

            if within_bounds(rotated_x_centered, rotated_y_centered, width, height):
                rotated_img.putpixel((rotated_x_centered, rotated_y_centered), data[w, h])

    return rotated_img
