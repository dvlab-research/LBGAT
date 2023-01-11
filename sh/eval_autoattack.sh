#!/bin/bash


### CIFAR100 models
#python auto_attack_eval.py --model-path pretrained_model/cifar100_lbgat0_wideresnet34-10-92.pt --mark cifar100_lbgat0_autoattack_widen10_seed0.txt --widen_factor 10 --num_classes 100 
#python auto_attack_eval.py --model-path pretrained_model/cifar100_lbgat6_wideresnet34-10.pt --mark cifar100_lbgat6_autoattack_widen10_seed0.txt --widen_factor 10 --num_classes 100 
#python auto_attack_eval.py --model-path pretrained_model/cifar100_lbgat0_wideresnet34-20-92.pt --mark cifar100_lbgat0_autoattack_widen20_seed0.txt --widen_factor 20 --num_classes 100 
#python auto_attack_eval.py --model-path pretrained_model/cifar100_lbgat6_wideresnet34-20-91.pt --mark cifar100_lbgat6_autoattack_widen20_seed0.txt --widen_factor 20 --num_classes 100 --dataparallel True 

python auto_attack_eval.py --model-path pretrained_model/cifar100_lbgat9_wideresnet34-10_eps8_new.pt --mark cifar100_lbgat9_autoattack_widen10_seed0.txt --widen_factor 10 --num_classes 100 --dataparallel True --epsilon 8.0



### CIFAR10 models
