from PIL import Image
from pathlib import Path

def make_image_grid(imgs, rows, cols, save_path):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    
    grid.save(save_path)   

def crop_image(imagePath, imageName, outputPath):
    # create the output directory if it does not exist
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    image = Image.open(imagePath + imageName)
    width, height = image.size
    image = crop_center(image, min(width,height), min(width,height))
    image = image.resize((params['image_size'],params['image_size']))
    path = Path(outputPath + imageName)
    image.save(path)

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))