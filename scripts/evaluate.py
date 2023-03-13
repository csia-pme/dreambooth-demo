from pytorch_fid import calculate_fid_given_paths
from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os
import yaml

params = yaml.safe_load(open("../params.yaml"))["infere"]

def calculate_fig() :
    real_images_path = "./data/prepared"
    generated_images_path = "/"

    fid_score = calculate_fid_given_paths([real_images_path, generated_images_path])

def generate_images() :

    final_model = './model'
    pipe = StableDiffusionPipeline.from_pretrained(final_model, torch_dtype=torch.float16).to('cuda')
    
    output_path = './images'

    if not os.path.exists(output_path) :
        os.makedirs(output_path)

    prompt = 'a photo of sks person'

    image = pipe(prompt , num_inference_steps=params['steps'], guidance_scale=params['guidance']).images[0]
    image.save(output_path + '/1.jpg')