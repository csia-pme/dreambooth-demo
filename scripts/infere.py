from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import os
import yaml

def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

params = yaml.safe_load(open("../params.yaml"))["infere"]

#  model path
model_path = './model'
output_path = './images'

if not os.path.exists(path) :
    os.makedirs(path)

pipe = StableDiffusionPipeline.from_pretrained(final_model, torch_dtype=torch.float16).to('cuda')

prompt = params['prompt']

images = pipe(prompt , num_inference_steps=params['steps'], guidance_scale=params['guidance'])

image_grid = make_image_grid(images, 1, params['number_images'])
   
image_grid.save(output_path + '/grid.jpg') 