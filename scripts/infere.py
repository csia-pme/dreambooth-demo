from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

def infereFromModelId(model_id, pipe) :

    subjectName = os.environ.get('SUBJECT_NAME')

    prompts = [
        'a photo of ' + subjectName, 
        'a photo of ' + subjectName + ' dressed in swiss traditional clothing',
        'a photo of ' + subjectName + ' dressed as a peaky blinder',
        'a photo of ' + subjectName + ' as the joker from batman',
        'a portrait of ' + subjectName + ' as a medieval knight in armor',
        'a portrait of ' + subjectName + ' as a lego character',
        'a painting of ' + subjectName + ' as a the king of france',
        'a painting of ' + subjectName + ' in the style of Gustave Klimt',
        'a painting of ' + subjectName + ' in the style of Edgar Degas',
        'a painting of ' + subjectName + ' in the style of Salvador Dali',
        'a painting of ' + subjectName + ' in the style of medieval religious art',
        'a painting of ' + subjectName + ' in the style of romanticism',
        'a painting of ' + subjectName + ' in the style of impressionism',
        'a painting of ' + subjectName + ' in the style of an art deco poster, framed',
        'a painting of ' + subjectName + ' in the style of Vincent van Gogh',
        'a pencil drawing of ' + subjectName + ' in the style of Leonardo da Vinci',
        'a painting of ' + subjectName + ' in the style of Katsushika Hokusai',
        'a drawing of ' + subjectName + ' in the manga style',
        'a drawing of ' + subjectName + ' in the manga dragon ball z style',
        'a drawing of ' + subjectName + ' in the manga naruto style',
        ]
    print('Model id: ' + model_id)
    
    iteration = ''.join(filter(str.isdigit, model_id));

    if iteration == '' :
        iteration = 'final'
        
    for prompt in prompts:
        image_name = prompt.replace(' ', '-')
        image_name = prompt.replace(',', '')
        image_name = image_name.replace(' person', '')


        path = '../images/' + os.environ.get('SUBJECT_NAME') + '/' + image_name;
        if not os.path.exists(path) :
            os.makedirs(path)

        image = pipe(prompt , num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(path + '/' + iteration + '.jpg')
        
        #image = pipe(prompt + subjectGender + cleanStyle , num_inference_steps=50, guidance_scale=7.5).images[0]
        #image.save(path + '/' + iteration + '-clean.png')
        #image2 = pipe(prompt + subjectGender + realisticStyle , num_inference_steps=50, guidance_scale=7.5).images[0]
        #image2.save(path + '/' + iteration + '-realistic.png')

        print('Image generated ' + image_name)

print('Starting inference for intermediate models ...')

# list all intermediate models saved
listOfIntermetiateModels = [f.path for f in os.scandir('../model/' + os.environ.get('SUBJECT_NAME')) if f.is_dir()]
print('PWD : ' + os.getcwd())
print('Folder in target directory for intermediate models : ')
print(listOfIntermetiateModels)

# intermediate models
for model_name in listOfIntermetiateModels :
        # if model_name contains "checkpoint" then it's an intermediate model infere from it
    if 'checkpoint' in model_name :
        print('Identified intermediate model: ' + model_name + ' infere from it...')

        unetFolder = model_name + '/unet';
        textEncoderFolder = model_name + '/text_encoder';

        #print('Working directors is: ' + os.getcwd())
        #print('List of files in current directory: ')
        #print(os.listdir())
        #print('List of files in dreambooth-api directory: ')
        #print(os.listdir('/builds/AdrienAllemand/dreambooth-api'))
        #print('List of files in dreambooth-api scripts directory: ')
        #print(os.listdir('/builds/AdrienAllemand/dreambooth-api/scripts'))
        #print('List of files in parent directory: ')
        #print(os.listdir('..'))
        #print('List of files in tony model directory : ')
        #print(os.listdir('/builds/AdrienAllemand/dreambooth-api/model/tony'))
        #print('List of files in tony checkpoint directory : ')
        #print(os.listdir('/builds/AdrienAllemand/dreambooth-api/model/tony/checkpoint-80'))
        #print('List of files in text encoder directory: ')
        #print(os.listdir(textEncoderFolder))
#
        # Load the pipeline with the same arguments (model, revision) that were used for training
        #model_id = "stabilityai/stable-diffusion-2"
        model_id = "runwayml/stable-diffusion-v1-5"
        
        unet = UNet2DConditionModel.from_pretrained(unetFolder )

        # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        text_encoder = CLIPTextModel.from_pretrained(textEncoderFolder)

        pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16).to("cuda")

        infereFromModelId(model_name, pipeline)

        print('Inference done!')

print('Starting inference for final models ...')

# final model
final_model = '../model/' + os.environ.get('SUBJECT_NAME')
pipe = StableDiffusionPipeline.from_pretrained(final_model, torch_dtype=torch.float16).to('cuda')
infereFromModelId(final_model, pipe)
