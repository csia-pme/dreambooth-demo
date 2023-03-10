import os
import yaml
from PIL import Image
from pathlib import Path

params = yaml.safe_load(open("params.yaml"))["prepare"]

def crop_image(imagePath, imageName, outputPath):

    # create the output directory if it does not exist
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    image = Image.open(imagePath + imageName)
    width, height = image.size
    image = crop_center(image, min(width,height), min(width,height))
    image = image.resize((params['size'],params['size']))
    path = Path(outputPath + imageName)
    image.save(path)

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

folder_dir = './data/images'


for image in os.listdir(folder_dir):
    # check if the image ends with jpg
    if (image.endswith(".jpeg") or image.endswith(".jpg")):
        crop_image(folder_dir + '/', image, './data/prepared/')