from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

def infereFromModelId(model_id, pipe) :

    cleanStyle = ', beautiful face, symmetrical, centered, dramatic angle, ornate, details, smooth, sharp focus, illustration, realistic, cinematic, 8k, award winning, rgb , unreal engine, octane render, cinematic light, depth of field, blur'
    realisticStyle = 'medium closeup photo, detailed (wrinkles, blemishes!, folds!, viens, pores!!, skin imperfections:1.1), (wearing sexy lingerie set details:1.1), highly detailed glossy eyes, (looking at the camera), specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, centered, Fujifilm XT3, crystal clear'

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
    print('Model id: ' + model_id)
    if model_id:
        iteration = ''.join(filter(str.isdigit, model_id));
    else :
        iteration = 'final'
        
    for prompt in prompts:
        image_name = prompt.replace(' ', '-')


        path = '../images/' + os.environ.get('SUBJECT_NAME') + '/' + image_name;
        if not os.path.exists(path) :
            os.makedirs(path)

        image = pipe(prompt + subjectGender + cleanStyle , num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(path + '/' + iteration + '-clean.png')
        image2 = pipe(prompt + subjectGender + realisticStyle , num_inference_steps=50, guidance_scale=7.5).images[0]
        image2.save(path + '/' + iteration + '-realistic.png')

        print('Image generated ' + image_name)

print('Starting inference...')

# final model
final_model = '../model/' + os.environ.get('SUBJECT_NAME')
pipe = StableDiffusionPipeline.from_pretrained(final_model, torch_dtype=torch.float16).to('cuda')
infereFromModelId(final_model, pipe)

# list all intermediate models saved
listOfIntermetiateModels = [f.path for f in os.scandir('../model/' + os.environ.get('SUBJECT_NAME')) if f.is_dir()]
print('PWD : ' + os.getcwd())
print('Folder in target directory for intermediate models : ' + listOfIntermetiateModels)

# intermediate models
for model_name in listOfIntermetiateModels :
        # if model_name contains "checkpoint" then it's an intermediate model infere from it
    if 'checkpoint' in model_name :
        print('Identified intermediate model: ' + model_name + ' infere from it...')
        # Load the pipeline with the same arguments (model, revision) that were used for training
        model_id = "runwayml/stable-diffusion-v1-5"

        unet = UNet2DConditionModel.from_pretrained(model_name + "/unet")

        # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        text_encoder = CLIPTextModel.from_pretrained(model_name + "/text_encoder")

        pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16).to("cuda")

        infereFromModelId(model_id, pipeline)

print('Inference done!')