from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from autoattack import AutoAttack

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--mark', default=None, type=str,
                            help='log file name')
parser.add_argument('--widen_factor', default=None, type=int,
                            help='widen_factor for wideresnet')
parser.add_argument('--num_classes', default=10, type=int,
                            help='cifar10 or cifar100')
parser.add_argument('--dataparallel', default=False, type=bool,
                            help='whether model is trained with dataparallel')


args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])

if args.num_classes == 100:
   testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
   test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
else:
   testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
   test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  X,
                  y,
                  adversary,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = adversary.run_standard_evaluation(X, y, bs=X.size(0))
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader, adverary):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, adverary)
        robust_err_total += err_robust
        natural_err_total += err_natural

    open("Logs/"+args.mark,"a+").write("robust_err_total: "+str(robust_err_total)+"\n")
    open("Logs/"+args.mark,"a+").write("natural_err_total: "+str(natural_err_total)+"\n")

def main():

    if args.white_box_attack:
        # white-box attack
        open(args.mark,"a+").write('pgd white-box attack\n')
        model = WideResNet(num_classes=args.num_classes, widen_factor=args.widen_factor)
        if args.dataparallel:
           model = nn.DataParallel(model).to(device)
        else:
           model = model.to(device)

        model.load_state_dict(torch.load(args.model_path))
        adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard', log_path = "Logs/"+args.mark)
        adversary.seed = 0
        eval_adv_test_whitebox(model, device, test_loader, adversary)
    

if __name__ == '__main__':
    main()
