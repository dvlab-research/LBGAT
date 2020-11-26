 
# Learnable Boundary Guided Adversarial Training (https://arxiv.org/pdf/2011.11164.pdf)
## Our method creates new state-of-the-art model robustness on CIFAR-100 while preserving the highest natural accuracy up to 2020/11/25!

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
Note: this is one partial list results for comparisons with methods without using additional data up to 2020/11/25. Full list can be found at https://github.com/fra31/auto-attack. 

| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) |  
| :---: | :---: | :---: | :---: | :---: |
| 1 | **LBGAT (Ours)**                                         | WRN-34-20 | 62.55 | **30.20** |   
| 2 | [(Gowal et al. 2020)](https://arxiv.org/abs/2010.03593)  | WRN-70-16 | 60.86 | 30.03 |
| 3 | **LBGAT (Ours)**                                         | WRN-34-10 | 60.64 | **29.33** |
| 4 | [(Wu et al. 2020)](https://arxiv.org/abs/2004.05884)     | WRN-34-10 | 60.38 | 28.86 |
| 5 | **LBGAT (Ours)**                                         | WRN-34-10 | **70.25** | 27.16 |
| 6 | [(Chen et al. 2020)](https://arxiv.org/abs/2010.01278)   | WRN-34-10 | 62.15 | 26.94 |
| 7 | **TRADES (alpha=6)**                                     | WRN-34-10 | 56.50 | 26.87 |
| 8 | [(Sitawarin et al. 2020)](https://arxiv.org/abs/2003.09347)  | WRN-34-10 | 62.82 | 24.57 |
| 9 | [(Rice et al. 2020)](https://arxiv.org/abs/2002.11569)       | RN-18     | 53.83 | 18.95 |


## CIFAR-10 L-inf
Note: this is one partial list results for comparisons with published methods without using additional data up to 2020/11/25. Full list can be found at https://github.com/fra31/auto-attack.

wait to update !!


