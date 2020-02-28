# Distributed Training of a Single Machine Learning Model

[Source: Chris & Antje from Amazon](https://master.d2j834wqg8s4j0.amplifyapp.com/intro.html)

Steps involved in Scaling the Training to Multiple Nodes

1. Setting up the containers to take care of the Hardware and Software Dependencies
  > Package a docker image and push it into a container registry. Which will then be pulled to create a cluster with uniform images.
2. Setting up the cloud compute and storage infrastructure management to perform the Training.

## Horovod brings the HPC Techniques to Deep Learning
[Source: Andrew Gibiansky](http://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)

Horovod architecture addresses the communication overhead during distributed training. It uses scatter-reduce and all-gather phases to average the gradients computed by different compute nodes and compute the new weights by applying the averaged gradient. The data transferred during these phases is independent of the number compute nodes used.

Data Transferred = <img src="https://render.githubusercontent.com/render/math?math=2(N-1)\frac{K}{N}">

Considering N compute nodes then each will send and receive the gradient values N-1 times for the scatter-reduce and N-1 times for the allgather phases. Where K represents the is total size of the gradient values. The above equation is independent of N.

## Prerequisites

### Launch Amazon SageMaker Notebook instance, Terminal and Download code

Open Amazon SageMaker Service -> Notebook instances then click on create notebook instance.
Fill up Notbook instance settings (name and instance type) and Permissions and encryption (IAM role and enable root access to the notebook) and click Create notebook instance. Open the instance and click on the IAM role ARN link to attach AdministratorAccess policy. Open the terminal using the files->New->Terminal from the newly created Jupyter instance.

```
cd ~/SageMaker
git clone https://github.com/data-science-on-aws/kubeflow.git
```

## Distributed Training with Amazon EKS and KubeFlow

Navigate to kubeflow -> notebooks -> part-3-kubernetes

docker directory contains docker files for the custom container image for training using Amazon EKS.
docker/code contains the training script files.
specs directory contains the list of YAML files to configure KubeFlow.

```
source ~/.bash_profile

cd ~/SageMaker/kubeflow/notebooks/part-3-kubernetes/

#### Install `eksctl`
# To get started we'll first install the `awscli` and `eksctl` CLI tools. [eksctl](https://eksctl.io) simplifies the process of creating EKS clusters.

pip install awscli --upgrade --user

curl --silent --location "https://github.com/weaveworks/eksctl/releases/download/latest_release/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp

sudo mv /tmp/eksctl /usr/local/bin

eksctl version
# stdout: [ℹ]  version.Info{BuiltAt:"", GitCommit:"", GitTag:"0.13.0"}

#### Install `kubectl`
# `kubectl` is a command line interface for running commands against Kubernetes clusters. 
# Run the following to install Kubectl

curl -o kubectl https://amazon-eks.s3-us-west-2.amazonaws.com/1.14.6/2019-08-22/bin/linux/amd64/kubectl

chmod +x ./kubectl

sudo mv ./kubectl /usr/local/bin

kubectl version --short --client
# stdout: Client Version: v1.14.7-eks-1861c5

#### Install `aws-iam-authenticator`

curl -o aws-iam-authenticator https://amazon-eks.s3-us-west-2.amazonaws.com/1.14.6/2019-08-22/bin/linux/amd64/aws-iam-authenticator

chmod +x ./aws-iam-authenticator

sudo mv aws-iam-authenticator /usr/local/bin

aws-iam-authenticator version
# {"Version":"v0.4.0","Commit":"c141eda34ad1b6b4d71056810951801348f8c367"}

#### Install jq and envsubst (from GNU gettext utilities) 
sudo yum -y install jq gettext

#### Verify the binaries are in the path and executable
for command in kubectl jq envsubst
  do
    which $command &>/dev/null && echo "$command in path" || echo "$command NOT FOUND"
  done
# stdout : kubectl in path
# jq in path
# envsubst in path

```

Setup the AWS_REGION and AWS_CLUSTER_NAME environment variables
```
export AWS_REGION=us-west-2
echo "export AWS_REGION=${AWS_REGION}" | tee -a ~/.bash_profile

export AWS_CLUSTER_NAME=kubeflowcluster
echo "export AWS_CLUSTER_NAME=${AWS_CLUSTER_NAME}" | tee -a ~/.bash_profile

```
### Create the EKS cluster
```
source ~/.bash_profile

eksctl create cluster \
    --name ${AWS_CLUSTER_NAME} \
    --version 1.14 \
    --region ${AWS_REGION} \
    --nodegroup-name cpu-nodes \
    --node-type c5.xlarge \
    --nodes 5 \
    --node-volume-size 100 \
    --node-zones us-west-2a \
    --timeout=40m \
    --zones=us-west-2a,us-west-2b,us-west-2c \
    --alb-ingress-access \
    --auto-kubeconfig

stdout:
[ℹ]  eksctl version 0.13.0
[ℹ]  using region us-west-2
[ℹ]  subnets for us-west-2a - public:192.168.0.0/19 private:192.168.96.0/19
[ℹ]  subnets for us-west-2b - public:192.168.32.0/19 private:192.168.128.0/19
[ℹ]  subnets for us-west-2c - public:192.168.64.0/19 private:192.168.160.0/19
[ℹ]  nodegroup "cpu-nodes" will use "ami-0c13bb9cbfd007e56" [AmazonLinux2/1.14]
[ℹ]  using Kubernetes version 1.14
[ℹ]  creating EKS cluster "kubeflowcluster" in "us-west-2" region with un-managednodes
[ℹ]  will create 2 separate CloudFormation stacks for cluster itself and the initial nodegroup
[ℹ]  if you encounter any issues, check CloudFormation console or try 'eksctl utils describe-stacks --region=us-west-2 --cluster=kubeflowcluster'
[ℹ]  CloudWatch logging will not be enabled for cluster "kubeflowcluster" in "us-west-2"
[ℹ]  you can enable it with 'eksctl utils update-cluster-logging --region=us-west-2 --cluster=kubeflowcluster'
[ℹ]  Kubernetes API endpoint access will use default of {publicAccess=true, privateAccess=false} for cluster "kubeflowcluster" in "us-west-2"
[ℹ]  2 sequential tasks: { create cluster control plane "kubeflowcluster", createnodegroup "cpu-nodes" }
[ℹ]  building cluster stack "eksctl-kubeflowcluster-cluster"
[ℹ]  deploying stack "eksctl-kubeflowcluster-cluster"
```




