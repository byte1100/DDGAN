import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
from torchvision import transforms
import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    return args_attack
args_attack = parse()


use_cuda=True
image_nc=3

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_20.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

tf = transforms.Compose([
    # transforms.CenterCrop(170),
    transforms.Resize(args_attack.global_settings.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image = Image.open(sys.argv[1])
img = image.convert("RGB")
img = tf(img).unsqueeze(0)

for i in range(1):
    img = img.to(device)
    img.cuda()
    perturbation = pretrained_G(img)
    #perturbation = torch.clamp(perturbation, -0.05, 0.05)
    perturbation = torch.clamp(perturbation, -0.06, 0.06)
    torch.save(perturbation, args_attack.global_settings.single_path)
    print(perturbation.size())


