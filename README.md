# dreambooth-exeriment

This is a simple experiment to MLOpsify the dreambooth project.

## Sources

https://huggingface.co/docs/diffusers/training/dreambooth
https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
https://blog.paperspace.com/dreambooth-stable-diffusion-tutorial-1/

## steps

I created a pod on the k8s cluster based on the following yaml file:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: gpu-container
    image: nvidia/cuda:12.0.1-runtime-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1
    command: ["/bin/bash"]
    args: ["-c", "while true; do echo 'Running GPU pod...'; sleep 30; done"]
  restartPolicy: Never
  ```

The next steps are run on the pod. To log onto the pod use :
This assumes you have configured your kubectl to target the right cluster.

```bash
kubectl exec -it gpu-pod --namespace=dreambooth-experience -- /bin/bash
```

We will need GIT and Python3 and pip3 to run the training
```bash
apt update
apt install git
apt install -y python3-pip
pip3 install --upgrade pip
````

Next steps follow the hugginface tutorial :
https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
that is a better version of :
https://huggingface.co/docs/diffusers/training/dreambooth

```bash
git clone https://github.com/huggingface/diffusers
pip install -U -r diffusers/examples/dreambooth/requirements.txt
accelerate config default
```

The I cloned this repository to have data 

```bash
git clone https://gitlab.com/AdrienAllemand/dreambooth-api.git
````

I used the personal access token in `./secrets` folder to clone the repo. Now I need to export config to run the training

// TODO update the training comment to use `./scripts/train.sh`
```bash
mkdir -p /model
mkdir -p /class
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/dreambooth-api/data/tony"
export OUTPUT_DIR="/model"
export CLASS_DIR="/class"
```

Finally TRAIN

```bash
accelerate launch /diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of tony" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```
```bash

## with prior preservation
  accelerate launch /diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of tony the person" \
  --class_prompt="a photo of a person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

after a few minutes the model training is done we can do some inference 

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of tony in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("tony-bucket.png")
```

to pull the image locally use

### prompt tests

This prompt worked very well:
> prompt = "A portrait of tony, beautiful, vivid colors, no default, symmetrical, centered, ornate, details, smooth, sharp focus, illustration, realistic, cinematic, artstation, award winning, unreal engine, octane render, cinematic light, depth of field, Blender and Photoshop, dynamic dramatic cinematic lighting, very inspirational"

> prompt = "A portrait of tony as a medieval knight, beautiful face, vivid colors, no default, symmetrical, centered, ornate, details, smooth, sharp focus, illustration, realistic, cinematic, artstation, award winning, unreal engine, octane render, cinematic light, depth of field, Blender and Photoshop, dynamic dramatic cinematic lighting, very inspirational"
>
## K8s gitlab runner

We want our gitlab pipeline to execute within the kubernetes cluster. To do so we need to install a gitlab runner on the cluster.

>See https://docs.gitlab.com/runner/install/kubernetes.html

>https://www.youtube.com/watch?v=0Fes86qtBSc

Using helm we install the runner on the cluster
```bash
helm repo add gitlab https://charts.gitlab.io
helm repo update
```

We need to create a configuration file for the runner :
This needs to be downloaded on the machine we use to administrate the K8s cluster.
```bash
wget https://gitlab.com/gitlab-org/charts/gitlab-runner/-/tree/main/values.yaml
```

Specify the gitlab instance url in the `values.yaml` configuration file. Here we use a http://gitlab.com but you could run your own gitlab and put it's url.
```yaml
# values.yaml
gitlabUrl: https://gitlab.com/
```
> if you get a 401 error when trying to register the runner, check that the url is correct and is using https.
> 
We also need to add  the runnerRegistrationToken. You can get it directly from your gitlab UI > Repository > Settings > CI/CD > Runners > Expand the runner > Copy the token. 

The mentioned video stores it directly in the `values.yaml` file, it's not a good practice to store secrets in the configuration file. We will use a kubernetes secret to store the token then update the `runners.secret` value in `values.yaml` with the name of the secret.

> See https://docs.gitlab.com/runner/install/kubernetes.html#store-registration-tokens-or-runner-tokens-in-secrets

To encode the token, use the following command :
```bash
echo -n 'my-token' | base64
````

Then paste the output un the secret definition file 'gitlab-runner-secret.yaml'. You can find a template for this file in the `./k8s` folder of this repository. __Do not save the secret file in the repository__.

Once the secret is created add the secret name to your `values.yaml` file.
```yaml
#[...]
runners:
  secret: gitlab-runner-secret
#[...]
```

Next step is to create a RBAC configuration to give permission to create pods to the cluster.
```yaml
#[...]
rbac:
  create: true # create a service account and a role binding
  rules:  # these are the default roles uncommented
    - resources: ["configmaps", "pods", "pods/attach", "secrets", "services"]
      verbs: ["get", "list", "watch", "create", "patch", "update", "delete"]
    - apiGroups: [""]
      resources: ["pods/exec"]
      verbs: ["create", "patch", "delete"]
#[...]
```


## Assign GPU to the pipeline pods
One problem we would encounter running the pipeline "as is" is that pods are deployed on gpuless nodes. We can fix this by adding a nodeSelector to the pipeline pods.

```yaml
runners:
  config: |
    [[runners]]
      [runners.kubernetes]
        namespace = "{{.Release.Namespace}}"
        image = "ubuntu:16.04"
      [runners.kubernetes.node_selector]
        "nvidia.com/gpu.present" = "true"
```
  
Finally we can install the runner on the cluster
```bash
helm upgrade --install --namespace dreambooth-experience gitlab-runner -f ./values.yaml gitlab/gitlab-runner
```
If you need to update an existing runner use 
```bash
helm upgrade --install --namespace dreambooth-experience gitlab-runner -f ./values.yaml gitlab/gitlab-runner

You should now see your runner in the gitlab UI > Repository > Settings > CI/CD > Runners, correctly registered as a Project Runner.

## Creating a pipeline

First, let's make a gitlab ci file to run our training script. We will use the nvidia/cuda:12.0.1-runtime-ubuntu22.04 image to have access to the GPU.

```yaml
stages:
  - train

variables:
  SUBJECT_NAME: tony  # the name of the person we want to train the model on


train-job:
  stage: train
  image: nvidia/cuda:12.0.1-runtime-ubuntu22.04
  script:
    - sh ./scripts/installation.sh
    - sh ./scripts/train.sh
    - python ./scripts/infere.py

```


### OLD
update config.toml on the runner to add the following lines

```sh
kubectl exec -it gitlab-runner-54dc59c857-fcqhg   --namespace=dreambooth-experience -- /bin/bash
```
```toml
nvidia.com/gpu.present=tru


### OLD
First let's save the command to run the runner

```bash
helm repo add gitlab https://charts.gitlab.io
helm repo update
helm upgrade --install agent-configuration gitlab/gitlab-agent \
    --namespace dreambooth-experiment \
    --create-namespace \
    --set image.tag=v15.9.0-rc1 \
    --set config.token=<your-agent-access-token-here> \
    --set config.kasAddress=wss://kas.gitlab.com
```