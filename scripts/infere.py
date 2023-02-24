from diffusers import StableDiffusionPipeline
import torch
import os

print('Starting inference...')

model_id = '../model/' + os.environ.get('SUBJECT_NAME')
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')

style = ' , beautiful face, symmetrical, centered, dramatic angle, ornate, details, smooth, sharp focus, illustration, realistic, cinematic, 8k, award winning, rgb , unreal engine, octane render, cinematic light, depth of field, blur'
subjectName = os.environ.get('SUBJECT_NAME')
subjectGender = ' ' + os.environ.get('SUBJECT_GENDER', '')
subjects = [
    'a portrait of ' + subjectName + '' , 
    'a portrait of ' + subjectName + ' as a medieval knight', 
    'a portrait of ' + subjectName + ' as Ironman helmetless',
    'a portrait of ' + subjectName + ' as a medieval swiss person',
    'a portrait of ' + subjectName + ' as a 17 century noble',
    'a portrait of ' + subjectName + ' as president of the USA',
    'a portrait of ' + subjectName + ' as an elf',
    ]

for subject in subjects:

    image_name = subject.replace(' ', '-') + '.png'
    print('Inference of ' + image_name)
    prompt = subject + subjectGender + style 
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save('../images/' + os.environ.get('SUBJECT_NAME') + '/' + image_name)

    print('Image generated ' + image_name)

print('Inference done!')