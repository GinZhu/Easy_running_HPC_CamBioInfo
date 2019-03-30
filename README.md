# Easy running HPC CamBioInfo
Example code and script for using Cam-HPC

# HPC and Account
## GPUs, RAM, and CPUs limitation
Cambridge HPC [Wilkes2](https://www.hpc.cam.ac.uk/systems/wilkes-2 "Wilkes2") provides 90 nodes (i.e. PCs) of GPUs to use, they are very powerful:
> Each node contains

> 1 x Intel Xeon E5-2650 v4 2.2GHz 12-core processor

> 96GB RAM

> Single Rail Mellanox EDR

> 4 x Nvidia P100 GPU 16GB.

> Theoretical node peak (CPU+GPU): 19.61 TFlops/s

> Connects to the 5PB/s LustreFS shared with Peta4.

Normally you will submit your job to a single node, because this is more convenient for coding. You can requist one or at most 4 GPUs to use for each node. For each GPU, 24GB RAM and 3 CPUs will be paired automatically. It is recommended to have a test on kiiara/crunchy/cozy of RAM and GPU usage before submitting jobs to HPC.

## Account
1. To use HPC, you need first creat an account in [Tier2 safe system](https://www.archer.ac.uk/tier2/ "Tier2 safe system")
2. In the page to update your personal details, upload the public key of your computer, remember that you can only login into HPC with the correct priviate key.  
3. At the top the page, 'login accounts' ->  'Request login accounts', choose the project
4. Send email to your supervisor for the permission.
5. It will normally take hours to creat your account in the system.** In the page of login accounts/your_user_name, if the status is pending, please wait.** 

## Login HPC (via multi computers)
To access the Wilkes2-GPU (GPU cluster) nodes:
> ssh your_user_name&#64;login-gpu.hpc.cam.ac.uk

If password is required, it means that you should check the public key you have uploaded.

If you want to login in the HPC system via more than one computers, for example, your PC in the lab and your personal Laptop, there are two possible ways to do this:
1. in Tier2 Update Personal Details page, change the ssh public key every time you change the computer to login.
2. You can add the public key(s) into .ssh/authorized_keys on hpc. **Caution: this may not be safe.**

# File system:
The file system is very similar as kiiara/crunchy/cozy, you will have your home folder (40GB), and local folder (shared with the whole project members).
1. To check how many free space do you have, using [quota](https://docs.hpc.cam.ac.uk/hpc/user-guide/io_management.html?highlight=quota "quota")
2. Your local folder is /home/user_name/rds/project_name/, please creat your own folder.
3. You can use scp to transfer files from/to HPC
> scp user_name&#64;login-gpu.hpc.cam.ac.uk:PATH_OF_FILE WHERE_TO_SAVE

> scp -r  user_name&#64;login-gpu.hpc.cam.ac.uk:PATH_OF_DIR WHERE_TO_SAVE

4. Using &#42; can copy all files with certain conditon:
> scp user_name&#64;login-gpu.hpc.cam.ac.uk:PATH/slurm* WHERE_TO_SAVE

# Using PyTorch:
1.  Using PyTroch: https://docs.hpc.cam.ac.uk/hpc/software-packages/pytorch.html?highlight=pytorch
2. In the above tutorial, [modules](https://docs.hpc.cam.ac.uk/hpc/user-guide/development.html?highlight=module "modules") are mentioned. I suggest to copy the following into ~/.bashrc:
> $ module load cuda/9.0 intel/mkl/2017.4

> $ module load python-3.6.1-gcc-5.4.0-64u3a4w py-numpy-1.12.1-gcc-5.4.0-cjrgw2k py-matplotlib-2.2.2-gcc-5.4.0-6oe6fph

> $ module load py-virtualenv-15.1.0-gcc-5.4.0-gu4wi6c

3. Please use Python 3 if you want to use TensorFlow on HPC, here is the [tutorial](https://docs.hpc.cam.ac.uk/hpc/software-packages/tensorflow.html "tutorial").

# Submitting Jobs:
You need to use slurm system to submit your job(s) to the queue.
1. First, modify the [example script](https://github.com/GinZhu/Easy_running_HPC_CamBioInfo/blob/master/hpc_start.script "example script") for your job. You may also use the offical [example](https://docs.hpc.cam.ac.uk/hpc/user-guide/batch.html "example")
2. submit the job by:
> sbatch your_script

3. In the script, you can set num of nodes, gpus, max running time, and so on. You can require at most 36 hours. You will also decide receive what emails. For detials, see [sbatch]( https://slurm.schedmd.com/sbatch.html "sbatch").
4. Remeber using the correct Python interpreter in your script, take the virtualenv as an example:
> Your_virtualenv/bin/python3

Tips: it is suggested to test and debug your code on kiiara/crunchy/cozy before submitting the job, because the average queue time is around 2-3 hours on HPC. Based on my experience, if your code works on kiiara/crunchy/cozy, it will work well with HPC.

# Using Multi-GPUs
Finally there is the chance to use multi-gpus with HPC.
1. Here is an [example](https://github.com/GinZhu/Easy_running_HPC_CamBioInfo/blob/master/hpc_run_parallelizing.py "example") to running multi training processing on multi GPUs.
2. Multi-gpus can also be used to train big neural networks (for example, ResNet), for PyTorch, using [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html "DataParallel").
3. PyTorch DataParallel cannot work well with BatchNorm currently, please use [Synchronized Batch Norm]( https://github.com/vacancy/Synchronized-BatchNorm-PyTorch "Synchronized Batch Norm").

# Official documents:
1. Using HPC: https://docs.hpc.cam.ac.uk/hpc/user-guide/batch.html
