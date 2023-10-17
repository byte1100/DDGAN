import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
import time
from tqdm import tqdm
from AttGAN.data import check_attribute_conflict
from Celeba256_gen import Generator
from Celeba256_dis import Discriminator1,Discriminator2
models_path = './models/'
os.environ ["CUDA_VISIBLE_DEVICES"] = "4"
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model_attentiongan,
                 model_stargan,
                 model_attgan,
                 attgan_args,
                 #model_hisd,
                 # model_num_labels,
                 up_attentiongan,
                 up_stargan,
                 up_attgan,
                 up_hisd,
                 image_nc,
                 box_min,
                 box_max,
                 E,
                 F,
                 T,
                 G,
                 reference):
        output_nc = image_nc
        self.device = device

        self.model_attentiongan = model_attentiongan
        self.model_stargan = model_stargan
        self.model_attgan = model_attgan
        self.attgan_args = attgan_args
        self.E = E
        self.F = F
        self.T = T
        self.G = G
        self.reference = reference
        #self.model_hisd= model_hisd

        self.up_attentiongan = up_attentiongan
        self.up_stargan = up_stargan
        self.up_attgan = up_attgan
        self.up_hisd = up_hisd

        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.up = None

        self.loss_fn = nn.MSELoss().to(device)

        # 这一部分是advGAN，需要改输入的维度、标签等，以及输出的损失（可能不用改）
        self.gen_input_nc = image_nc
        # 普通GAN结构
        # self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        # self.netDisc1 = models.Discriminator1(image_nc).to(device)
        # self.netDisc2 = models.Discriminator2(image_nc).to(device)
        # transformer_GAN结构
        self.netG = Generator().to(device)
        self.netDisc1 = Discriminator1().to(device)
        self.netDisc2 = Discriminator2().to(device)


        self.netG.apply(weights_init)
        self.netDisc1.apply(weights_init)
        self.netDisc2.apply(weights_init)
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.0005)
        self.optimizer_D1 = torch.optim.Adam(self.netDisc1.parameters(),
                                            lr=0.0001)
        self.optimizer_D2 = torch.optim.Adam(self.netDisc2.parameters(),
                                            lr=0.0001)


        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, idx, x, c_org,att_b_list):
        # optimize D1
        for i in range(1):
            perturbation = self.netG(x)
            #adv_images = torch.clamp(perturbation, -0.06, 0.06) + x
            adv_images = perturbation + x
            up = self.up_attentiongan + self.up_stargan

            self.optimizer_D1.zero_grad()
            pred_real = self.netDisc1(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real,device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc1(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake,device=self.device))
            loss_D_fake.backward(retain_graph=True)
            loss_D_GAN = loss_D_real +loss_D_fake
            self.optimizer_D1.step()

        # optimize D2
        for i in range(1):
            #u
            attention_adv = self.model_attentiongan.generator_images(x, c_org, self.up_attentiongan)
            star_adv = self.model_stargan.generator_images(x, c_org, self.up_stargan)
            attention_fake = self.model_attentiongan.generator_images(x, c_org, perturbation)
            star_fake = self.model_stargan.generator_images(x, c_org, perturbation)

            #att对抗真假图片
            att_adv, att_fake = [], []
            for i, att_b in enumerate(att_b_list):
                att_b_ = (att_b * 2 - 1) * self.attgan_args.thres_int
                if i > 0:
                    att_b_[..., i - 1] = att_b_[..., i - 1] * self.attgan_args.test_int / self.attgan_args.thres_int
                with torch.no_grad():
                    gen = self.model_attgan.G(x + self.up_attgan, att_b_)
                    gen_noattack = self.model_attgan.G(x + perturbation, att_b_)
                att_adv.append(gen)
                att_fake.append(gen_noattack)

            # loss_D2_real = 0.0
            # for j in range(len(attention_adv)):
            #     # real_loss
            #     pred_real_attention = self.netDisc2(attention_adv[j])
            #     pred_real_star = self.netDisc2(star_adv[j])
            #     pred_real = pred_real_attention + 0.5 * pred_real_star
            #     loss_D2_real += F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            # # loss_att_real = 0.0
            # # for k in range(len(att_adv)):
            # #     pred_real_att = self.netDisc2(att_adv[k])
            # #     loss_att_real += F.mse_loss(pred_real_att, torch.ones_like(pred_real, device=self.device))
            # # loss_D2_real = loss_att_real/len(att_adv)
            # loss_D2_real = loss_D2_real / len(attention_adv)
            # loss_D2_real.backward()
            # 
            # loss_D2_fake  = 0.0
            # for j in range(len(attention_adv)):
            #     # fake_loss
            #     pred_fake_attention = self.netDisc2(attention_fake[j])
            #     pred_fake_star = self.netDisc2(star_fake[j])
            #     pred_fake = pred_fake_attention +0.5 * pred_fake_star
            #     loss_D2_fake += F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            # # loss_att_fake = 0.0
            # # for k in range(len(att_fake)):
            # #     pred_fake_att = self.netDisc2(att_fake[k])
            # #     loss_att_fake += F.mse_loss(pred_fake_att, torch.ones_like(pred_real, device=self.device))
            # loss_D2_fake = loss_att_fake/len(att_adv)
                # clean
            self.optimizer_D2.zero_grad()
            c = self.E(x + self.up_hisd)
            c_trg = c
            s_trg = self.F(self.reference, 1)
            c_trg = self.T(c_trg, s_trg, 1)
            gen_noattack = self.G(c_trg)
            # adv
            c = self.E(adv_images)
            c_trg = c
            s_trg = self.F(self.reference, 1)
            c_trg = self.T(c_trg, s_trg, 1)
            gen = self.G(c_trg)

            pred_real_hisd = self.netDisc2(gen_noattack)
            loss_D2_real_hisd = F.mse_loss(pred_real_hisd, torch.ones_like(pred_real_hisd, device=self.device))
            pred_real_attentiongan = self.netDisc2(attention_adv[0])
            loss_D2_real_attentiongan = F.mse_loss(pred_real_attentiongan, torch.ones_like(pred_real_attentiongan, device=self.device))
            pred_real_stargan = self.netDisc2(star_adv[0])
            loss_D2_real_stargan = F.mse_loss(pred_real_stargan, torch.ones_like(pred_real_stargan, device=self.device))
            pred_real_attgan = self.netDisc2(att_adv[0])
            loss_D2_real_attgan = F.mse_loss(pred_real_attgan, torch.ones_like(pred_real_stargan, device=self.device))
            loss_D2_real = loss_D2_real_hisd + loss_D2_real_attentiongan + loss_D2_real_stargan + loss_D2_real_attgan
            loss_D2_real.backward()

            pred_fake_hisd = self.netDisc2(gen)
            loss_D2_fake_hisd = F.mse_loss(pred_fake_hisd, torch.zeros_like(pred_real_hisd, device=self.device))
            pred_fake_attention = self.netDisc2(attention_fake[0])
            loss_D2_fake_attentiongan = F.mse_loss(pred_fake_attention, torch.zeros_like(pred_fake_attention, device=self.device))
            pred_fake_stargan = self.netDisc2(star_fake[0])
            loss_D2_fake_stargan = F.mse_loss(pred_fake_stargan, torch.zeros_like(pred_fake_stargan, device=self.device))
            pred_fake_attgan = self.netDisc2(att_fake[0])
            loss_D2_fake_attgan = F.mse_loss(pred_fake_attgan, torch.zeros_like(pred_fake_attgan, device=self.device))
            loss_D2_fake = loss_D2_fake_hisd + loss_D2_fake_attentiongan + loss_D2_fake_stargan + loss_D2_fake_attgan
            loss_D2_fake.backward(retain_graph=True )

            loss_D2_GAN = loss_D2_real + loss_D2_fake
            self.optimizer_D2.step()
        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()
            # pred_fake = self.netDisc1(adv_images)
            # loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake,device=self.device))
            # loss_G_fake.backward(retain_graph=True)
            #
            # pred_fake = self.netDisc1(adv_images)
            # loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake,device=self.device))
            # loss_G_fake.backward(retain_graph=True)
            self.optimizer_G.zero_grad()
            G_loss_GAN_D1 = loss_D_fake
            G_loss_GAN_D2 = loss_D2_fake
            loss_adv = G_loss_GAN_D1 + G_loss_GAN_D2

            # 水印的范数
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

            adv_lambda = 100
            pert_lambda = 1
            if loss_perturb.item() >= 20:
                loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            else:
                loss_G = adv_lambda * loss_adv + 0.01 * loss_perturb
            #loss_G = pert_lambda * loss_perturb
            loss_G.backward()
            print(loss_adv)
            print(loss_perturb)
            print(loss_G)
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_D_fake.item(), loss_perturb.item(), loss_adv.item(),perturbation

    def train(self, attack_dataloader, epochs,args_attack):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D1 = torch.optim.Adam(self.netDisc1.parameters(),
                                                    lr=0.00001)
                self.optimizer_D2 = torch.optim.Adam(self.netDisc2.parameters(),
                                                    lr=0.00001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.000001)
                self.optimizer_D1 = torch.optim.Adam(self.netDisc1.parameters(),
                                                    lr=0.000001)
                self.optimizer_D2 = torch.optim.Adam(self.netDisc2.parameters(),
                                                    lr=0.000001)

            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            up = None
            # 这里改改输出输出
            for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
                if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
                #if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == (args_attack.global_settings.num_test * 5):
                    break
                img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
                att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
                att_a = att_a.type(torch.float)
                #用于AttGAN的标签
                att_b_list = [att_a]
                for i in range(self.attgan_args.n_attrs):
                    tmp = att_a.clone()
                    tmp[:, i] = 1 - tmp[:, i]
                    tmp = check_attribute_conflict(tmp, self.attgan_args.attrs[i], self.attgan_args.attrs)
                    att_b_list.append(tmp)


                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, up = \
                    self.train_batch(idx,img_a,c_org,att_b_list)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(attack_dataloader)
            num_batch = 8
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))
            self.up = up
            torch.save(self.up, args_attack.global_settings.mean_path)
            # save generator
            if epoch%1==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

