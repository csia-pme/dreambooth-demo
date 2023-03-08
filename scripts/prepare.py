import os
import sys
import yaml
from PIL import Image

params = yaml.safe_load(open("../params.yaml"))["prepare"]

def crop_image(imagePath):
    image = Image.open(imagePath)
    width, height = image.size
    image = crop_center(image, min(width,height), min(width,height))
    image = image.resize((params['size'],params['size']))
    image.save(imagePath)

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

subjectName = sys.argv[1]

if(subjectName is None):
    print ('Subject is not defined')
    exit(1)

folder_dir = '../data/' + subjectName
for image in os.listdir(folder_dir):

    # check if the image ends with png
    if (image.endswith(".jpeg") or image.endswith(".jpg")):
        crop_image(folder_dir + '/' + image)