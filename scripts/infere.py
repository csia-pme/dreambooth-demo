from diffusers import StableDiffusionPipeline
import torch
import os

def infereFromModelId(model_id) :

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')

    cleanStyle = ', beautiful face, symmetrical, centered, dramatic angle, ornate, details, smooth, sharp focus, illustration, realistic, cinematic, 8k, award winning, rgb , unreal engine, octane render, cinematic light, depth of field, blur'
    realisticStyle = 'medium closeup photo, detailed (wrinkles, blemishes!, folds!, moles, viens, pores!!, skin imperfections:1.1), (wearing sexy lingerie set details:1.1), highly detailed glossy eyes, (looking at the camera), specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, centered, Fujifilm XT3, crystal clear'

    subjectName = os.environ.get('SUBJECT_NAME')
    subjectGender = ' ' + os.environ.get('SUBJECT_GENDER', '')

    prompts = [
        'a photo of ' + subjectName, 
        'a photo of ' + subjectName + ' as a medieval knight',
        'a photo of ' + subjectName + ' as a medieval swiss person',
        'a photo of ' + subjectName + ' as a 17 century noble',
        'a photo of ' + subjectName + ' as president of the USA',
        'a photo of ' + subjectName + ' as an elf'
        ]

    for prompt in prompts:

        image_name = prompt.replace(' ', '-')
        image = pipe(prompt + subjectGender + cleanStyle , num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save('../images/' 
            + os.environ.get('SUBJECT_NAME') + '/' 
            + image_name + '/'
            + int(''.join(filter(str.isdigit, model_id))) + '-clean.png')
        image2 = pipe(prompt + subjectGender + realisticStyle , num_inference_steps=50, guidance_scale=7.5).images[0]
        image2.save('../images/' 
            + os.environ.get('SUBJECT_NAME') + '/' 
            + image_name + '/'
            + int(''.join(filter(str.isdigit, model_id))) + '-realistic.png')


        print('Image generated ' + image_name)



print('Starting inference...')

# list all intermediate models saved
listOfIntermetiateModels = [f.path for f in os.scandir('../model/' + os.environ.get('SUBJECT_NAME')) if f.is_dir()]

# intermediate models
for model_name in listOfIntermetiateModels :
    # if model_name contains "checkpoint" then it's an intermediate model infere from it
    if 'checkpoint' in model_name :
        infereFromModelId(model_name);      

# final model
infereFromModelId('../model/' + os.environ.get('SUBJECT_NAME'))


print('Inference done!')