# DreamBooth Experiment

This is a simple experiment to MLOpsify the DreamBooth project using DVC and CML.

- [DreamBooth Experiment](#dreambooth-experiment)
  - [The Experiment](#the-experiment)
  - [Pre-requisites](#pre-requisites)
  - [Setup](#setup)
    - [Configure Kubernetes](#configure-kubernetes)
      - [Add KubeConfig](#add-kubeconfig)
      - [Create a Namespace](#create-a-namespace)
      - [Configure the Namespace](#configure-the-namespace)
    - [Configure MinIO](#configure-minio)
    - [Configure MiniKube](#configure-minikube)
  - [Run the Experiment on the Cluster Manually](#run-the-experiment-on-the-cluster-manually)
    - [Connect to the Environment](#connect-to-the-environment)
      - [Create the K8s Pod](#create-the-k8s-pod)
      - [Log onto the Pod](#log-onto-the-pod)
    - [Clone the Repository](#clone-the-repository)
    - [Installation](#installation)
      - [Clone the Diffusers Repo](#clone-the-diffusers-repo)
      - [Install Python Requirements](#install-python-requirements)
      - [Install JQ and YQ](#install-jq-and-yq)
    - [Pull the Data with DVC](#pull-the-data-with-dvc)
    - [Run the Experiment](#run-the-experiment)
    - [Push the Results](#push-the-results)
  - [Integrate with GitLab](#integrate-with-gitlab)
    - [K8s GitLab Runner Setup](#k8s-gitlab-runner-setup)
    - [Kubernetes Runner Configuration](#kubernetes-runner-configuration)
      - [Assign GPU to the Pipeline Pods](#assign-gpu-to-the-pipeline-pods)
  - [Integrate with GitHub](#integrate-with-github)
    - [Setup Runner](#setup-runner)
      - [Install cert-manager in your cluster](#install-cert-manager-in-your-cluster)
      - [Generate a GitHub Personal Access Token (PAT)](#generate-a-github-personal-access-token-pat)
      - [Configure ARC](#configure-arc)
      - [Configure PAT as a Secret in your Cluster](#configure-pat-as-a-secret-in-your-cluster)
      - [Deploy the GitHub Self-hosted Runner](#deploy-the-github-self-hosted-runner)
    - [Verify Workflows](#verify-workflows)
  - [Multi-GPU Checkpoint infernce](#multi-gpu-checkpoint-infernce)
  - [Create the DVC Pipeline](#create-the-dvc-pipeline)
    - [Add Preparation Stage](#add-preparation-stage)
    - [Add Train Stage](#add-train-stage)
    - [Add infernce Stage](#add-infernce-stage)
  - [Clean up](#clean-up)
  - [Resources](#resources)
  - [Contributing](#contributing)
    - [Markdown Linting and Formatting](#markdown-linting-and-formatting)

## The Experiment

This experiment is based on the DreamBooth project.

The goal is to train a model to generate images of a given subject. The model is trained on a dataset of images of the subject and a dataset of images of people in general. The model is then used to generate images of the subject.

<div class="center">
<h3><center> Experiment diagram</center> </h3>
<center>

```mermaid
graph TD
A[Subject images] --> B[Fine tune the model using dreambooth]
A2[Stable diffusion 1.5] --> B
B --> C[Use the new model to create images of the subject]
```

</center> 
</div>

## Pre-requisites

- VPN connection to the IICT network
- Access to the IICT Kubernetes cluster (https://rancher.iict.ch/)
- Access to IICT MinIo (https://minio-aii.iict.ch)
- Kubernetes CLI ([`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl/))

## Setup

### Configure Kubernetes

#### Add KubeConfig

You first need to add the Kubernetes config file to your local machine. You can do this by following these steps :

- Login to [Rancher](https://rancher.iict.ch/dashboard/)
- Select the `iict` cluster
- On the top right corner click on **Download KubeConfig** button
- Move the downloaded file to `~/.kube/config`

> **WARNING :** The KubeConfig file is a sensitive file. You should not share it or commit it to any public repository.

#### Create a Namespace

To create a namespace for this experiment you can run the following command :

```bash
kubectl create namespace <your namespace name>
```

> **Note :** Replace `<your namespace name>` with the name of the namespace you want to create.

#### Configure the Namespace

Finally, set the namespace as the default namespace for your current context :

```bash
kubectl config set-context --current --namespace=<your namespace name>
```

> **Note :** Replace `<your namespace name>` with the name of the namespace you created at the previous step.

Verify that the namespace is set as the default namespace for your current context :

```bash
kubectl config view --minify | grep namespace:
```

### Configure MinIO

You will need to configure MinIO to be able to access the data. To do this you can follow these steps :

- Login to [MinIO](https://console-minio-aii.iict.ch/login)
- On the sidebar, select **Buckets**
- Click on **Create a new bucket**
- Enter the name of the bucket you want to create
- Click on **Create Bucket**

### Configure MiniKube

If you wish to run the experiment locally you can use MiniKube.

You can install it by following the instructions [here](https://minikube.sigs.k8s.io/docs/start/).

Once installed you can start a cluster with the following command :

```bash
minikube start
```

> **Tip :** You can specify the number of CPUs and the amount of RAM you want to allocate to the cluster by using the `--cpus` and `--memory` flags.

This will start a local cluster and MiniKube will configure your `~/.kube/config` file to use the "minikube" cluster and "default" namespace.

All of the commands in this tutorial are the same for MiniKube and the IICT cluster.

If you would like to switch back to the IICT cluster you can run the following command :

```bash
kubectl config use-context iict --namespace <your namespace name>
```

## Run the Experiment on the Cluster Manually

As this experiment requires a lot of VRAM we recommend you run it on a GPU enabled machine with more than 24Go of VRAM. We used 2x Nvidia A40 GPUs with 48 Go of VRAM but you can probably get away with less if you edit the training script to use less VRAM. (see https://github.com/huggingface/diffusers/tree/main/examples/dreambooth for more details on possible configurations of the training script)

> You can get a look at the `gitlab-ci.yml` file to see the pipeline execution steps.

> You will need the full content of this repository to run the experiment

<div class="center">
<h3>
  <center>
    Full run of the experiment
  </center>
</h3>
<center>

```mermaid
graph TD
B[Clone the rep\n <i>git clone</i>]
B --> C[Install the requirements\n <i>scripts/installation.sh</i>]
C --> D[pull the data\n <i>dvc pull</i>]
D -- manual --> E[Prepare the data \n <i>scripts/prepare.sh</i>]
E --> F[Train the model\n <i>scripts/train.sh</i>]
F --> G[infer images \n <i>scripts/infernce.sh</i>]
G -- if in CI --> H[Report the results \n <i>scripts/report.sh</i>]
D -- with dvc --> I[Reproduce the experiment \n <i>dvc repro</i>]
I -- if in CI ---> H
```

  </center> 
</div>

### Connect to the Environment

Depending on your context, you will need to connect to the environment you will use to run this experiment.

If you want to run the experiment on a Kubernetes pod in your cluster you can do the following :

#### Create the K8s Pod

> You will need to have a Kubernetes cluster with GPUs available to run this experiment.

> You will need to have `kubectl` configured to target the right cluster or use the web interface rancher to create the pod : https://rancher.iict.ch

> For our experiment we run the pod in a namespace called `dreambooth-experience`.

You can either use the command line version or go through [Rancher](https://rancher.iict.ch/dashboard/) web interface to create the pod. Our rancher is only available when connect to the VPN, being connected to the Wifi of the school is not enough.

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
      command: ['/bin/bash']
      args: ['-c', "while true; do echo 'Running GPU pod...'; sleep 30; done"]
  restartPolicy: Never
```

Note that we use the container image `nvidia/cuda:12.0.1-runtime-ubuntu22.04` to have a pre-configured environment with CUDA.

#### Log onto the Pod

> This assumes you have configured your `kubectl` to target the right cluster pod and it has access to GPUs.

> Our pod is named `gpu-pod` and is in the `dreambooth-experience` namespace.

```bash
kubectl exec -it gpu-pod -- /bin/bash
```

Congratulation, you now have a shell on a GPU enabled machine in our K8s cluster. If you want to be sure you have access to the GPU from the pod you created you can run the following command :

```bash
nvidia-smi
```

You should see something like this :

```bash
root@gpu-pod:/# nvidia-smi
Wed Mar 22 07:48:09 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A40          On   | 00000000:CA:00.0 Off |                    0 |
|  0%   32C    P8    29W / 300W |      0MiB / 46068MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Clone the Repository

> You need to have Git installed: `apt install -y git`

> You need to have a personal access token to clone the repo. You should make your own and have the possibility to save it in a file in the `./secrets` folder if you want for further use, the folder should be ignored by Git at all time.

```bash
git clone https://github.com/csia-pme/dreambooth-example-with-mlops
```

Most of the following steps are run from the root of the repo.

### Installation

The `installation.sh` script will install all the requirements to run the experiment in a Debian based environment such as the k8s pod we created earlier.

> **Note :** This will not create a virtual environment, it will install the requirements globally on .

If you want to install the requirements manually, you can do the following :

#### Clone the Diffusers Repo

> **Note :** You need to have Git installed

The diffusers are used to train the model. We will use the DreamBooth example to fine tune the model.

```bash
git clone https://github.com/huggingface/diffusers
```

#### Install Python Requirements

> You need to have python3 in version 3.10 and pip3 installed

```bash
# Create the virtual environment
python3 -m venv .venv
# Activate the virtual environment
. .venv/bin/activate
# Install our requirements
pip3 install --upgrade pip
pip3 install -r requirements.txt
# Install the diffusers requirements
pip install -e ./diffusers
pip install -r ./diffusers/examples/dreambooth/requirements.txt
```

#### Install JQ and YQ

To give access to the parameters in `params.yaml` to our `sh` scripts, we use `yq` over `jq`. `jq` is a command line JSON processor. `yq` is a wrapper around `jq` that allows you to read and write yaml files.

You can see an example of reading a parameter in `scripts/train.sh` using `yq` :

```bash
MODEL_NAME=$(yq -r '.train.model_name' params.yaml)
...
```

This is used to centralize changes to the parameters in one file without needing to modify the experiment code.

```bash
apt install -y jq
jq --version
pip install yq
yq --version
```

Parameters can be found in `params.yaml` in the root of the repo.

### Pull the Data with DVC

When you installed the dependencies from our `requirement.txt` file you should have installed DVC. DVC is used to manage the data used in the experiment.

DVC uses S3 to store the data. For each data tracked and stored by DVC on S3, metadata about their state are tracked using the `dvc.lock` and `.dvc` files. The metadata file is used to check if the data has changed and if it needs to be updated when dvc needs to access it (eg. when using `dvc repro` it can skip steps if all their dependencies are unchanged by using their cached output).

> **Note :** Git is in charge of tracking the metadata files so you should commit the metadata files but not the data itself.

We first initialize DVC in the repo :

```bash
dvc init
```

> **Note :** If DVC was already initialized, you can use `dvc init -f` to force reinitialize it.

This will create a `.dvc` folder in the root of the repo.

We will use an S3 self-hosted by a MinIO service to store the data. To configure DVC to use it, we need to create a `~/.dvc/config` file.

You can create the file with the following command :

```bash
dvc remote add myremote s3://<your bucket name> && \
    dvc remote modify myremote endpointurl <your minio url>
```

> **Note :** Replace `<your bucket name>` with the name of the bucket you want to use and `<your minio url>` with the url of your minio instance.

Next, you can add the MinIO credentials to the DVC config with the following command :

```bash
echo -n 'MinIO S3 Secret Access Key : ' && \
    read -s MINIO_SECRET_ACCESS_KEY && \
    dvc remote modify --local myremote access_key_id <your minio user> && \
    dvc remote modify --local myremote secret_access_key $MINIO_SECRET_ACCESS_KEY && \
    unset MINIO_SECRET_ACCESS_KEY && \
    echo -e '\nAdded MinIO credentials to DVC config'
```

> **Note :** Replace `<your minio user>` with the user you want to use to access the bucket.

> **WARNING :** You should not store secrets in the `~/.dvc/config` file. You should use the `--local` flag to store the secret in the local config file. This file is ignored by Git. See the [DVC config documentation](https://dvc.org/doc/command-reference/config#description) for more information.

To pull the data, we use the following command :

```bash
dvc pull
```

Depending on the state of the experiment on the S3 bucket, this command can take a while to complete.

### Run the Experiment

You now have the latest "state" of the experiment both for the code and the data. You can now run the experiment.

You can run the experiment using DVC. It is an abstraction of the three stages `prepre`, `train` and `infer`. Or you can run the stages individually. Depending on the state of the data you pulled from the S3 bucket, DVC might skip some stages as their dependencies are unchanged. If you want to force the execution of a stage you can use the `--force` flag or to force the execution of all stages you can use the `--force-all` flag.

To run all at once with DVC you can do :

```bash
dvc repro
 # or
dvc repro --force-all
```

### Push the Results

Once the experiment is done, you can push the results to the S3 bucket using the following command :

```bash
dvc push
git add -A
git commit -m "Experiment results"
git push
```

## Integrate with GitLab

### K8s GitLab Runner Setup

We want our GitLab pipeline to execute within the Kubernetes cluster. To do so we need to install a GitLab runner on the cluster. Note that what we install is not the final pod used to execute our pipelines but a pod that will spawn other pods to execute the pipelines as needed.

> See https://docs.gitlab.com/runner/install/kubernetes.html

> Gitlab runner on kubernetes tutorial : https://www.youtube.com/watch?v=0Fes86qtBSc

Using helm we install the gitlab runner on the cluster

```bash
helm repo add gitlab https://charts.gitlab.io
helm repo update
```

We need to create a configuration file for the runner :

This needs to be downloaded on the machine we use to administrate the K8s cluster.

```bash
wget https://gitlab.com/gitlab-org/charts/gitlab-runner/-/tree/main/values.yaml
```

Specify the GitLab instance url in the `values.yaml` configuration file. Here we use a http://gitlab.com but you could run your own GitLab and put it's url.

```yaml
# values.yaml
gitlabUrl: https://gitlab.com/
```

> if you get a 401 error when trying to register the runner, check that the url is correct and is using https.

> We also need to add the runnerRegistrationToken. You can get it directly from your gitlab UI > Repository > Settings > CI/CD > Runners > Expand the runner > Copy the token.

The mentioned video stores it directly in the `values.yaml` file, it's not a good practice to store secrets in the configuration file. We will use a Kubernetes secret to store the token then update the `runners.secret` value in `values.yaml` with the name of the secret.

> See https://docs.gitlab.com/runner/install/kubernetes.html#store-registration-tokens-or-runner-tokens-in-secrets

To encode the token, use the following command :

```bash
echo -n 'my-token' | base64
```

Then paste the output un the secret definition file 'gitlab-runner-secret.yaml'. You can find a template for this file in the `./k8s` folder of this repository. **Do not save the secret file in the repository**.

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
  rules: # these are the default roles uncommented
    - resources: ['configmaps', 'pods', 'pods/attach', 'secrets', 'services']
      verbs: ['get', 'list', 'watch', 'create', 'patch', 'update', 'delete']
    - apiGroups: ['']
      resources: ['pods/exec']
      verbs: ['create', 'patch', 'delete']
#[...]
```

### Kubernetes Runner Configuration

#### Assign GPU to the Pipeline Pods

One problem we would encounter running the pipeline "as is" is that pods are deployed on GPU-less nodes by the runner. We can fix this by adding a nodeSelector to the pipeline pods.

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
helm upgrade --install --namespace <your namespace name> gitlab-runner -f ./values.yaml gitlab/gitlab-runner
```

If you need to update an existing runner use

```bash
helm upgrade --install --namespace <your namespace name> gitlab-runner -f ./values.yaml gitlab/gitlab-runner
```

> **Note :** Replace `<your namespace name>` with the name of the namespace you created earlier.

You should now see your runner in the GitLab UI > Repository > Settings > CI/CD > Runners, correctly registered as a Project Runner.

## Integrate with GitHub

### Setup Runner

Let's start by installing the GitHub runner on the cluster. You can find the documentation of ARC [here](https://github.com/actions/actions-runner-controller).

> **Note : ** you need to choose what level you want to register your runner against

You can register it at a repository level or at an organization level. If you choose to register it at an organization level, you will need to specify the repository name in the `spec.repository` field of the runner, if you choose to register it at a organization level, you will need to specify the organization name in the `spec.organization` field of the runner. This is done in the 'github-runner-deployment.yaml' file.

#### Install cert-manager in your cluster

The cert-manager is a Kubernetes add-on to automate the management and issuance of TLS certificates from various issuing sources. It will be used to generate a TLS certificate for the GitHub runner. This is mandatory when using a self-hosted ARC runner.

For more information, see "[cert-manager](https://cert-manager.io/docs/installation/)."

> **Note :** This command uses v1.11.0 of cert-manager. Please replace with a later version, if available.

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.11.0/cert-manager.yaml
```

#### Generate a GitHub Personal Access Token (PAT)

Select the `repo` scope (Full control).

For more information, see "[Creating a personal access token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token)."

> Note you will need extra permissions if your runner is registered at the organization level such as `admin:org`, `admin:org_hook`, `notifications`, `read:public_key`, `read:repo_hook`, `repo, workflow`

#### Configure ARC

To configure ARC, run the following command :

```bash
helm repo add actions-runner-controller https://actions-runner-controller.github.io/actions-runner-controller
```

#### Configure PAT as a Secret in your Cluster

Run the following command to configure the GitHub PAT as a secret in the `actions-runner-system` namespace.

```bash
echo -n 'Enter the GitHub PAT : ' && \
    read -s GITHUB_PAT && \
    helm upgrade --install --namespace actions-runner-system --create-namespace \
    --set=authSecret.create=true \
    --set=authSecret.github_token=$GITHUB_PAT \
    --wait actions-runner-controller actions-runner-controller/actions-runner-controller && \
    unset GITHUB_PAT
```

#### Deploy the GitHub Self-hosted Runner

The GitHub runner configuration is stored at `k8s/github-runner-deployment.yaml` file.

It creates/updates a Service Account, associated Role and a RunnerDeployment ensuring there is always a runner listening for a pipeline job.

> **Note :** If you want to use your own repo url, update the `k8s/github-runner-deployment.yaml` file with your repository url.

Apply the configuration to the cluster.

```bash
kubectl apply -f k8s/github-runner-deployment.yaml
```

### Verify Workflows

To verify that the setup was successful, you can run the following commands :

```bash
$ kubectl get runners
NAME                               REPOSITORY                               STATUS
github-custom-runner-cst5x-6268k   csia-pme/dreambooth-example-with-mlops   Running

$ kubectl get pods
NAME                               READY   STATUS    RESTARTS   AGE
github-custom-runner-cst5x-6268k   2/2     Running   0          1m
```

Congratulation, you have a runner waiting for a job ! Trigger your pipeline to see it in action.

## Multi-GPU Checkpoint infernce

There is a problem with accelerate when generating checkpoints in a multi-GPU architecture. There are 2 solutions : run without checkpointing or run with a single GPU.

To run with a single GPU, we need to modify the following line to the `train.sh` script :

```bash
accelerate launch --num_processes=1 --gpu_ids=0 ./diffusers/examples/dreambooth/train_dreambooth.py \
...
```

Otherwise, the output of the checkpoint will be lacking the `unet/` folder and it will be impossible to make images from that checkpoint.

## Create the DVC Pipeline

This has already been done in this repository. You can find the pipeline in the `./dvc` folder. The pipeline is composed of 3 stages : `prepare`, `train` and `infer`. The following is a description of each stage and the commands used to create them.

### Add Preparation Stage

This stage will take images in the `data/images` folder and prepare them for training. It will crop them to the size specified in the `prepare.size` parameter (in pixel). The output of this stage will be in the `data/prepared` folder.

- Stage name :
  - `prepare`
- Parameters :
  - `prepare.size`
- Dependencies :
  - `scripts/prepare.py`
  - `data/images`
- Outputs :
  - `data/prepared`
- CMD to run :
  - `python3 scripts/prepare.py`

```bash
# Add the prepare stage to the DVC pipeline
dvc stage add -n prepare \
  -p prepare.size \
  -d scripts/prepare.py \
  -d data/images \
  -o data/prepared \
  python3 scripts/prepare.py
```

### Add Train Stage

This stage will take the prepared images and train the model. The output of this stage will be in the `model/` folder.

```bash
# Add the train stage to the DVC pipeline
dvc stage add -n train \
  -p train.model_name \
  -p train.instance_prompt \
  -p train.class_prompt \
  -p train.image_size \
  -p train.learning_rate \
  -p train.steps \
  -d scripts/train.sh \
  -d data/prepared \
  -o model \
  sh scripts/train.sh
```

### Add infernce Stage

This stage will take the trained model and generate images from it based on a prompt defined in the params.yaml file. The output of this stage will be in the `/images` folder.

```bash
dvc stage add -n infer \
  -p infer.prompt \
  -p infer.guidance \
  -p infer.infer_seed \
  -p infer.number_images \
  -p infer.steps \
  -d scripts/infer.py \
  -d models \
  -o images \
  python3 scripts/infer.py
```

## Clean up

To clean up the resources created in this tutorial, run the following commands:

```bash
# Delete the GitHub Self-hosted Runners
kubectl delete -f k8s/github-runner-deployment.yaml
# Delete the ARC deployment
helm uninstall actions-runner-controller --namespace actions-runner-system
# Delete the cert-manager namespace
kubectl delete namespace cert-manager
# Delete the actions-runner-system namespace
kubectl delete namespace actions-runner-system
```

If you used minikube to create the cluster, you can delete the cluster by running the following command :

```bash
minikube delete --all
```

## Resources

- **DreamBooth fine-tuning example**
  https://huggingface.co/docs/diffusers/training/dreambooth

- **DreamBooth training example**
  https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

- **Stable Diffusion Tutorial**
  https://blog.paperspace.com/dreambooth-stable-diffusion-tutorial-1/

## Contributing

### Markdown Linting and Formatting

This repository uses the following VSCode:

- [`spell-right`](https://marketplace.visualstudio.com/items?itemName=ban.spellright) for spell checking
- [`prettier`](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode) for formatting markdown files.
