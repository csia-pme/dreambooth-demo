from diffusers import StableDiffusionPipeline
import torch

print('Starting inference...')

model_id = "../model/$SUBJECT_NAME"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')

style = ', beautiful face, symmetrical, centered, dramatic angle, ornate, details, smooth, sharp focus, illustration, realistic, cinematic, artstation, award winning, rgb , unreal engine, octane render, cinematic light, macro, depth of field, blur, red light and clouds from the back, highly detailed epic cinematic concept art CG render made in Maya, Blender and Photoshop, octane render, excellent composition, dynamic dramatic cinematic lighting, aesthetic, very inspirational, arthouse by Henri Cartier Bresson'
subjectName = os.environ.get('SUBJECT_NAME')
subjectGender = os.environ.get('SUBJECT_GENDER', '')
subjects = [
    'a portrait of ' + subjectName + ' ' , 
    'a portrait of ' + subjectName + ' as a medieval knight ', 
    'a portrait of ' + subjectName + ' as Ironman ',
    'a portrait of ' + subjectName + ' as a medieval swiss ',
    'a portrait of ' + subjectName + ' as a 17 century noble ',
    'a portrait of ' + subjectName + ' as president of the USA ',
    'a portrait of ' + subjectName + ' as an elf ',
    ]

for subject in subjects:
    prompt = subject + subjectGender + style 
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    image_name = subject.replace(' ', '-') + '.png'
    image.save(image_name)

    print('Image generated ' + image_name)

print('Inference done!')