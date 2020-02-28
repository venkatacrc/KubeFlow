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





