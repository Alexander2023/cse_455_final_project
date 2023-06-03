from PIL import Image
from random import random

def convert_grayscale_to_rgb(img: Image.Image):
    return Image.merge('RGB', (img, img, img))

def random_horizontal_flip(img: Image.Image, prob_flip: float):
    if random() > prob_flip:
        return img

    width, height = img.size
    data = img.load()

    for h in range(height):
        for w in range(width):
            data[w, h], data[width - 1 - w, h] = data[width - 1 - w, h], data[w, h]

    return img

def random_crop(img: Image.Image):
    pass

def random_rotation(img: Image.Image):
    pass
