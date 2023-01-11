 
# Learnable Boundary Guided Adversarial Training 
This repository contains the implementation code for the ICCV2021 paper:  
**Learnable Boundary Guided Adversarial Training** (https://arxiv.org/pdf/2011.11164.pdf)


# Updates: Training with Epsilon 8/255 on CIFAR-100
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | AWP                                                  | WRN-34-10 | **60.38** | **28.86** | - |
| 2 | LBGAT-AWP                                            | WRN-34-10 | **62.31** | **30.44** | - | 
| 3 | LBGAT-AWP*                                           | WRN-34-10 | **62.99** | **31.20** | [model](https://drive.google.com/file/d/1g8M77pgE7fCe9KzqRBUkexBN-6mcHkV-/view?usp=share_link)  |  


# Overview
In this paper, we proposed the "Learnable Boundary Guided Adversarial Training" to preserve high natural accuracy while enjoy strong robustness for deep models. An interesting phenomenon in our exploration shows that natural classifier boundary can benefit model robustness to some degree, which is different from the previous work that the improved robustness is at cost of performance degradation on natural data. **Our method creates new state-of-the-art model robustness on CIFAR-100 without extra real or Synthetic data under [auto-attack benchmark](https://robustbench.github.io/)**. 

![image](https://github.com/FPNAS/LBGAT/blob/main/assets/lbgat.jpg)

## Results and Pretrained models

  `  
Models are evaluated under the strongest AutoAttack(https://github.com/fra31/auto-attack) with epsilon 0.031.

Our CIFAR-100 models:  
[CIFAR-100-LBGAT0-wideresnet-34-10](https://drive.google.com/file/d/1CijxcgW1U8yfrB3n4dyxUbcotVaxyZyA/view?usp=sharing) 70.25 vs 27.16                                             
[CIFAR-100-LBGAT6-wideresnet-34-10](https://drive.google.com/file/d/1pzheoiTtoh0qKWcyjFwwxFK6GL0yXAQI/view?usp=sharing) 60.64 vs 29.33    
[CIFAR-100-LBGAT6-wideresnet-34-20](https://drive.google.com/file/d/18iond836snl_chrBL0s7f_BY-5AUWfej/view?usp=sharing) 62.55 vs 30.20    


Our CIFAR-10 models:  
[CIFAR-10-LBGAT0-wideresnet-34-10](https://drive.google.com/file/d/1JufuOi5szINv2oSZ7iNnFrKzNaXiLG1-/view?usp=sharing) 88.22 vs 52.86  
[CIFAR-10-LBGAT0-wideresnet-34-20](https://drive.google.com/file/d/1RpqN3QwD7-QNIFGQfqcFG9FOQTAg4LOm/view?usp=sharing) 88.70 vs 53.57  

  
## CIFAR-100 L-inf 
Note: this is one partial results list for comparisons with methods without using additional data up to 2020/11/25. Full list can be found at https://github.com/fra31/auto-attack. TRADES (alpha=6) is trained with official open-source code at https://github.com/yaodongyu/TRADES.  

| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) |  
| :---: | :---: | :---: | :---: | :---: |
| 1 | **LBGAT (Ours)**                                         | WRN-34-20 | **62.55** | **30.20** |   
| 2 | [(Gowal et al. 2020)](https://arxiv.org/abs/2010.03593)  | WRN-70-16 | 60.86 | 30.03 |
| 3 | **LBGAT (Ours)**                                         | WRN-34-10 | 60.64 | **29.33** |
| 4 | [(Wu et al. 2020)](https://arxiv.org/abs/2004.05884)     | WRN-34-10 | 60.38 | 28.86 |
| 5 | **LBGAT (Ours)**                                         | WRN-34-10 | **70.25** | 27.16 |
| 6 | [(Chen et al. 2020)](https://arxiv.org/abs/2010.01278)   | WRN-34-10 | 62.15 | 26.94 |
| 7 | [(Zhang et al. 2019)](https://arxiv.org/abs/1901.08573) **TRADES (alpha=6)**                                     | WRN-34-10 | 56.50 | 26.87 |
| 8 | [(Sitawarin et al. 2020)](https://arxiv.org/abs/2003.09347)                                                      | WRN-34-10 | 62.82 | 24.57 |
| 9 | [(Rice et al. 2020)](https://arxiv.org/abs/2002.11569)                                                           | RN-18     | 53.83 | 18.95 |


## CIFAR-10 L-inf
Note: this is one partial results list for comparisons with **previous published methods** without using additional data up to 2020/11/25. Full list can be found at https://github.com/fra31/auto-attack. TRADES (alpha=6) is trained with official open-source code at https://github.com/yaodongyu/TRADES. “*” denotes methods aiming to speed up adversarial training. 

| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) |  
| :---: | :---: | :---: | :---: | :---: |
| 1 | **LBGAT (Ours)**                                         | WRN-34-20 | **88.70** | **53.57** |   
| 2 | [(Zhang et al.)]()                                       | WRN-34-10 | 84.52	| 53.51 |
| 3 | [(Rice et al. 2020)]()                                   | WRN-34-20 | 85.34	|	53.42 |
| 4 | **LBGAT (Ours)**                                         | WRN-34-10 | **88.22** | 52.86 |
| 5 | [(Qin et al., 2019)]()                        	          | WRN-40-8	 | 86.28	|	52.84 |
| 6 | [(Zhang et al. 2019)](https://arxiv.org/abs/1901.08573) **TRADES (alpha=6)**                                     | WRN-34-10 | 84.92 | 52.64 |
| 7 | [(Chen et al., 2020b)]()                                 |	WRN-34-10	| 85.32	| 51.12 |
| 8 | [(Sitawarin et al., 2020)]()                 	           | WRN-34-10	| 86.84	| 50.72 |
| 9 | [(Engstrom et al., 2019)]()	                             | RN-50	    | 87.03 |	49.25 |
| 10 | [(Kumari et al., 2019)]()             	                 | WRN-34-10	| 87.80	| 49.12 |
| 11 | [(Mao et al., 2019)]()	                                 |	WRN-34-10	| 86.21	| 47.41 |
| 12 | [(Zhang et al., 2019a)]()	                              | WRN-34-10	| 87.20	| 44.83 |
| 13 | [(Madry et al., 2018)]() **AT**              	          | WRN-34-10	| 87.14	| 44.04 |
| 14 | [(Shafahi et al., 2019)]()*                     	       | WRN-34-10	| 86.11	| 41.47 |
| 14 | [(Wang & Zhang, 2019)]()*	                              |	WRN-28-10	| **92.80**	| 29.35 |


# Get Started

Befor the training, please create the directory 'Logs' via the command 'mkdir Logs'.

## Training
```
bash sh/train_lbgat0_cifar100.sh
```

## Evaluation
before running the evaluation, please download the pretrained model.
```
bash sh/eval_autoattack.sh
```

# Acknowledgements
This code is partly based on the [TRADES](https://github.com/yaodongyu/TRADES) and [autoattack](https://github.com/fra31/auto-attack).

# Contact
If you have any questions, feel free to contact us through email (jiequancui@link.cuhk.edu.hk) or Github issues. Enjoy!

# BibTex
If you find this code or idea useful, please consider citing our work:
```
@inproceedings{cui2021learnable,
  title={Learnable boundary guided adversarial training},
  author={Cui, Jiequan and Liu, Shu and Wang, Liwei and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15721--15730},
  year={2021}
}
```  

