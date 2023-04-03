import os
import yaml
from utils import crop_image, make_image_grid

params = yaml.safe_load(open("params.yaml"))

# the folder where the images are stored
folder_dir = './data/images'

# loop through all the images in the folder a d prepare them
for image in os.listdir(folder_dir):
    # check if the image ends with jpg
    if (image.endswith(".jpeg") or image.endswith(".jpg")):
        crop_image(folder_dir + '/', image, './data/prepared/')

# Make a grid image from the prepared images to be used in the report
make_image_grid('./data/prepared/', 1, len(os.listdir('./data/prepared/')), './data/source_images_grid.jpg')