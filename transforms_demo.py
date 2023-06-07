from PIL import Image

from custom_transforms import random_horizontal_flip, random_rotation

original_image = Image.open("./demo_im.jpg")
flipped_image = random_horizontal_flip(original_image, 1.0)
rotated_image = random_rotation(original_image, 1.0, 0.349066)

original_image.show()
flipped_image.show()
rotated_image.show()
