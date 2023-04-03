from diffusers import StableDiffusionPipeline
import torch
import os
import yaml
from utils import make_image_grid

params = yaml.safe_load(open("./params.yaml"))["infer"]

#  model path
model_path = './models'
output_path = './images'

if not os.path.exists(output_path) :
    os.makedirs(output_path)

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')

prompts = [params['prompt']] * params['number_images']
print(prompts)

generator = torch.Generator("cuda").manual_seed(params['infer_seed'])
images = pipe(prompts, generator=generator, num_infernce_steps=params['steps'], guidance_scale=params['guidance']).images

make_image_grid(images, 1, params['number_images'], output_path + '/grid.jpg')
   