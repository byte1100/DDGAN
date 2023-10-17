import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
# 后加的
import json
import argparse
import copy
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm


import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F

from AttGAN.data import check_attribute_conflict



from data import CelebA
import attacks

from model_data_prepare import prepare
from evaluate import evaluate_multiple_models
os.environ ["CUDA_VISIBLE_DEVICES"] = "4"
def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    return args_attack

args_attack = parse()
print(args_attack)
os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
print("experiment dir is created")
os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path, 'results{}/setting.json'.format(args_attack.attacks.momentum))))
print("experiment config is saved")

if args_attack.global_settings.universal_perturbation_path:
   # pgd_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)
    up_attgan = torch.load(args_attack.global_settings.attgan_perturbation_path)
    up_stargan = torch.load(args_attack.global_settings.stargan_perturbation_path)
    up_attentiongan = torch.load(args_attack.global_settings.attentiongan_perturbation_path)
    up_hisd = torch.load(args_attack.global_settings.hisd_perturbation_path)



use_cuda=True
# 通道数
image_nc=3
epochs = 80
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


# 初始化四个模型，但是后面做的话，先测试attgan和stargan
attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()
print("finished init the attacked models")
#输出一些参数，关键的是模型model和水印up
advGAN = AdvGAN_Attack(device,
                          attentiongan_solver,
                          solver,
                          attgan,
                          attgan_args,
                          up_attentiongan,
                          up_stargan,
                          up_attgan,
                          up_hisd,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX,
                          E , F, T, G,reference)

advGAN.train(attack_dataloader, epochs,args_attack)
