# CML Runner

In this chapter we want to delegate the gpu enabled runner instantiation to CML.

We want our infrastructure to work in the following way :

```mermaid
sequenceDiagram
box local
participant User
end
box Github
participant Workflow
end
box Kubernetes
participant Self-Hosted-Runner
participant Self-Hosted-Runner-Pod
participant CML-Runner
participant CML-Runner-Pod
end

note over User: User pushes code to Github
User->>+Workflow: Trigger workflow
Self-Hosted-Runner->>Workflow: Listens for jobs
Workflow->>+Self-Hosted-Runner: Run workflow step 1
note over Self-Hosted-Runner: uses: iterative/setup-cml@v1 <br/>name: Deploy runner on k8s <br/> run: cml runner launch
Self-Hosted-Runner->>+CML-Runner: Creates a Github runner
CML-Runner->>-Self-Hosted-Runner: Created
Self-Hosted-Runner->>-Workflow: End step 1
CML-Runner->>Workflow: Listens for jobs

Workflow->>+CML-Runner: Run workflow step 2
CML-Runner->>+CML-Runner-Pod: Run workflow step 2
note over CML-Runner-Pod: image: nvidia/cuda[...] <br/>run: dvc pull <br/> run: dvc repro
CML-Runner-Pod->>-CML-Runner: End step 2
CML-Runner->>-Workflow: End step 2

```

## Sources

[Cml self-hosted kubernetes runners documentation](https://cml.dev/doc/self-hosted-runners?tab=Kubernetes)

## Modifications needed to our CI

A new step is added to our CI workflow to deploy the runner on kubernetes.

```yaml
launch-runner:
  runs-on:
    group: default
    labels: [self-hosted]
  container:
    image: ubuntu:latest
  steps:
    - uses: actions/setup-node@v3
      with:
        node-version: '16'
    - uses: actions/checkout@v3
    - uses: iterative/setup-cml@v1
    - name: Deploy runner on self-hosted k8s
      env:
        REPO_TOKEN: ${{ secrets.CML_PAT_TOKEN }}
        KUBERNETES_CONFIGURATION: ${{ secrets.KUBERNETES_CONFIGURATION }}
      run: |
        cml runner launch \
            --cloud=kubernetes \
            --labels=cml-gpu
```

> The runner used to run this step must have access to the kubernetes cluster to create the runner pod. As such we use our own self-hosted runner to execute this step.

Note we added a kubeconfig secret to our repository secrets. To view your own kubeconfig file (if it is configured locally) you can run the following command:

```bash
cat ~/.kube/config
```
