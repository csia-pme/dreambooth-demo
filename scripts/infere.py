from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

params = yaml.safe_load(open("params.yaml"))["prepare"]

def infereFromModelId(model_id, pipe) :

    subjectName = os.environ.get('SUBJECT_NAME')
    subjectGender = os.environ.get('SUBJECT_GENDER', '')
    if subjectGender != '' :
        subjectGender = ' ' + subjectGender

    prompts = [
        'a photo of ' + subjectName + subjectGender + ' as a medieval knight in armor',
        'a painting of ' + subjectName + subjectGender + ' as a the king of france',
        'a painting of ' + subjectName + subjectGender + ' in the style of Gustave Klimt',
        'a painting of ' + subjectName + subjectGender + ' in the style of Edgar Degas',
        'a painting of ' + subjectName + subjectGender + ' in the style of Salvador Dali',
        'a painting of ' + subjectName + subjectGender + ' by Andy Warhol',
        'a painting of ' + subjectName + subjectGender + ' by Salvador Dali',
        'a painting of ' + subjectName + subjectGender + ' in the style of mediaeval book illumination',
        'a painting of ' + subjectName + subjectGender + ' in the style of romanticism',
        'a painting of ' + subjectName + subjectGender + ' in the style of an art deco poster, framed',
        'a painting of ' + subjectName + subjectGender + ' in the style of Vincent van Gogh',
        'a painting of ' + subjectName + subjectGender + ' by Katsushika Hokusai',
        'a pencil drawing of ' + subjectName + subjectGender + ' in the style Leonardo da Vinci',
        ]
    print('Model id: ' + model_id)
    
    iteration = ''.join(filter(str.isdigit, model_id));

    if iteration == '' :
        iteration = 'final'
        
    for prompt in prompts:
        #remove generic keywords image name
        image_name = prompt.replace(' ', '-')
        image_name = image_name.replace(',', '')
        image_name = image_name.replace(' person', '')
        image_name = image_name.replace(' woman', '')


        path = '../images/' + os.environ.get('SUBJECT_NAME') + '/' + image_name;

        if not os.path.exists(path) :
            os.makedirs(path)


        image = pipe(prompt , num_inference_steps=params[steps], guidance_scale=params[guidance]).images[0]
        image.save(path + '/' + iteration + '.jpg')
        
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
