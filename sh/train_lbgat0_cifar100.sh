#!/bin/bash
#SBATCH --job-name=adv
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=cifar100_aodt6_train.txt
#SBATCH --gres=gpu:1
#SBATCH -c 2 
#SBATCH -A leojia
#SBATCH -p leojia
#SBATCH --qos leojia
#SBATCH -w proj78 

## Below is the commands to run , for this example,
## Create a sample helloworld.py and Run the sample python file 
## Result are stored at your defined --output location

source ~/jqcui/ENV/.bashrc
source /mnt/backup2/home/sliu/jqcui/ENV/py3.6pt1.2/bin/activate
python train_lbgat_cifar100.py -mark cifar100_lbgat0_0.031 -teacher_model ResNet18_cifar --beta 0 
