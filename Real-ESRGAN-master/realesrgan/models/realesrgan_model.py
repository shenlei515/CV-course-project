from copy import deepcopy
import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from zmq import device
from models.CMDSR import *
from models.Task_Contrastive_Loss import TaskContrastiveLoss
from basicsr.utils.registry import ARCH_REGISTRY
from models.utils import custom_collate,CustomDataset
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm


# class New_conv(nn.modules.conv.Conv2d):
#
#     def __init__(self,ori_model):
#         super(New_conv, self).__init__(ori_model)
#         self.weight=ori_model.weight
#         self.bias=ori_model.bias
#
#     def forward(self, input, condition_feature):
#         b, c, h, w = input.shape
#         weight = self.weight.unsqueeze(0) * self.scale * condition_feature
#         weight =weight.view(b*self.in_channel, self.out_channel, self.kernel_size, self.kernel_size)
#         input = input.view(1, b*self.in_channel, h, w)
#         bias = torch.repeat_interleave(self.bias, repeats=b, dim=0)
#         out = F.conv2d(input,weight,bias=bias,stride=self.stride,padding=self.padding,groups=b)
#         _, _, height, width = out.shape
#         out = out.view(b, self.out_channel, height, width)
#         return out


# @MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        print(issubclass(torch.nn.Module, SRGANModel)) # RealESRGANModel不是Module的子类
        self.__name__='RealESRGANModel'
        super(RealESRGANModel, self).__init__(opt)
        # print("using device===============",torch.cuda.current_device())
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        self.coefficient_contrastive_l1 = opt['coefficient_contrastive_l1']
        # add conditionNet
        self.n_block = 3
        self.n_conv_each_block = 1
        self.conv_index = '22'
        self.channels = 64
        self.batch_size=2
        self.support_size = self.batch_size
        self.Condition = ConditionNet(n_block=self.n_block, n_conv_each_block=self.n_conv_each_block,
                                      conv_index=self.conv_index,
                                      sr_in_channel=self.channels, support_size=self.support_size).to(self.device)
        print("what lr_condition++++++++++++++++++++++++++++++\
              "
              , float(opt['lr_condition']))
        self.conditionnet_mix_optimizer=torch.optim.Adam(self.Condition.parameters(), lr = float(opt['lr_condition']))
        self.conditionnet_mix_scheduler=torch.optim.lr_scheduler.MultiStepLR(self.conditionnet_mix_optimizer, milestones=opt['milestones'], gamma=float(opt['lr_gamma_condition']))

        # self.train_optimizer = torch.optim.Adam(self.parameters(), lr = float(opt['lr_condition']))
        # self.train_scheduler=torch.optim.lr_scheduler.MultiStepLR(train_optimizer, milestones=args.milestones, gamma=float(opt['lr_gamma']))
        # self.train_scheduler=torch.optim.lr_scheduler.MultiStepLR(self.train_optimizer, gamma=float(opt['lr_gamma']))


        self.Modulation = Modulations(n_block=self.n_block, n_conv_each_block=self.n_conv_each_block, conv_index=self.conv_index,
                                      sr_in_channel=self.channels).to(self.device)
        self.support_pool={}
        self.hr={}
        self.kind={}
        print(self.__class__.__bases__)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
                kind='up'
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
                kind='down'
            else:
                scale = 1
                kind = 'eq'
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                kind+='_gauss'
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
                kind+='_poisson'
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
                kind+='_up'
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                kind+='_down'
            else:
                scale = 1
                kind+='_eq'
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                kind+='_gauss'
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
                kind+='_poisson'

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                kind+='_inter'
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                kind+='_jpeg'

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
            data['lq']=self.lq
            # print(data['lq'])

            return kind
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm

        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        # self.output = self.net_g(self.lq)
        self.output= self.forward(self.lq, self.support_pool[self.kind[self.lq]])

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # toDo: contrastive loss


            l_g_total.backward()
            self.optimizer_g.step()

            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(gan_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def forward(self, X, support_set):
        condition_feature=self.Condition(support_set)
        condition_feature=self.Modulation(condition_feature)
        b,_,_,_=X.shape
        # print("condition_feature input to forward====",condition_feature.shape)
        condition_feature = torch.repeat_interleave(condition_feature, repeats=b//condition_feature.shape[2], dim=2)#[n_block, n_conv_each_block, batch, 1, 64, 1, 1]
        # print("condition_feature after interleave====",condition_feature.shape)
        condition_feature=torch.permute(condition_feature,[2,1,0,3,4,5,6])
        # print("condition_feature in forward====",condition_feature.shape)
        # condition_feature=condition_feature.reshape(-1,condition_feature.shape[-5],condition_feature.shape[-4],condition_feature.shape[-3],condition_feature.shape[-2],condition_feature.shape[-1])
        # condition_feature=condition_feature.reshape(condition_feature.shape[-5],condition_feature.shape[-4],condition_feature.shape[-3],condition_feature.shape[-2],condition_feature.shape[-1])
        # batch=X.shape[0]
        # condition_feature=torch.repeat_interleave(condition_feature, repeats=batch//condition_feature.shape[1], dim=1)
        
        # print("condition_feature.shape======",condition_feature.shape)
        # print("model_type==========",type(self.net_g))
        # print("model_class==========",self.net_g)
        # print("X.shape======",X.shape)

        # for _,layer in enumerate(self.net_g.modules()):
        #     if type(layer)==torch.nn.modules.conv.Conv2d and layer.weight.shape[-3]==64:
        #     # if layer=='Conv2d':
        #         # layer=New_conv(layer)
        #         # print("before Condition", layer.weight.shape)
        #         # layer.weight=torch.nn.Parameter(layer.weight.unsqueeze(0)*condition_feature[index])
        #         # layer.weight=torch.nn.Parameter(layer.weight.reshape(batch*layer.weight.shape[1], layer.weight.shape[-3], layer.weight.shape[-2], layer.weight.shape[-1]))
        #         # layer.bias = nn.Parameter(torch.repeat_interleave(layer.bias, repeats=batch, dim=0))
        #         # print("after Condition", layer.weight.shape)
        #         index+=1
        #     else:
        #         pass
        res=self.net_g([X, condition_feature])
        # print("The shape of output of Net::::::::",res.shape)
        return res

    @torch.no_grad()
    def prepare_supportset(self, train_data):
        # print("I'm using this Model")
        self.is_train=True
        # create LR image and support set
        index=self.feed_data(train_data)
        self.kind[self.lq]=index
        if index in self.support_pool.keys():
            torch.stack([self.support_pool[index], self.lq])
        else:
            self.support_pool[index]=self.lq
        if index in self.hr.keys():
            torch.stack([self.hr[index], self.gt])
        else:
            self.hr[index]=self.gt
        # print(self.support_pool)
        self.is_train=False

    def train_Condition(self):
        lr_double_patchs=CustomDataset(self.support_pool)
        hr_double_patchs=CustomDataset(self.hr)
        # dataset = torch.utils.data.TensorDataset(lr_double_patchs, hr_double_patchs)
        # print("lr_double_patchs.shape========", lr_double_patchs.shape)
        # print("hr_double_patchs.shape========", hr_double_patchs.shape)

        lr_loader=torch.utils.data.DataLoader(lr_double_patchs, batch_size=2*self.batch_size, shuffle=False)
        hr_loader=torch.utils.data.DataLoader(hr_double_patchs, batch_size=2*self.batch_size, shuffle=False)

        for _,patch in tqdm(enumerate(zip(lr_loader,hr_loader)),total=len(lr_loader),desc='inner_loop:', colour='green',leave=True):
            lr_double_patchs, hr_double_patchs=patch
            # print("lr_double_patchs===", lr_double_patchs.shape)
            # print("hr_double_patchs===", hr_double_patchs.shape)
            lr_double_patchs=torch.permute(lr_double_patchs,[1,0,2,3,4])
            hr_double_patchs = torch.permute(hr_double_patchs, [1,0,2,3,4])
            # print("lr_double_patchs.shape=======",lr_double_patchs.shape)
            # print("hr_double_patchs.shape=======",hr_double_patchs.shape)
            task_size=lr_double_patchs.shape[0]
            half_support_x1 = deepcopy(lr_double_patchs[:,0:lr_double_patchs.shape[1]//2,:,:,:]) # 将数据集分为两半
            half_support_x2 = deepcopy(lr_double_patchs[:,lr_double_patchs.shape[1]//2:,:,:,:])
            half_support_x1 = half_support_x1.reshape((task_size,-1,half_support_x1.shape[3],half_support_x1.shape[4])).to(self.device) # 将数据集整理成[k,3*n,w,h]
            # print("half_support_x1.shape====",half_support_x1.shape)
            half_support_x2 = half_support_x2.reshape((task_size,-1,half_support_x2.shape[3],half_support_x2.shape[4])).to(self.device)

            Contrastive_loss = TaskContrastiveLoss()
            train_pixel_loss = torch.nn.L1Loss()

            Condition=torch.nn.DataParallel(self.Condition)
            half_condition_feature1 = Condition(half_support_x1)
            half_condition_feature2 = Condition(half_support_x2)

            # print(half_condition_feature1)
            # print(half_condition_feature1)

            contrastive_loss, inner_class_distance, cross_class_distance = Contrastive_loss(half_condition_feature1, half_condition_feature2)

            
            #cal_sr_loss
            hr_support_x1 = deepcopy(hr_double_patchs[:,0:hr_double_patchs.shape[1]//2,:,:,:]).reshape((-1,3,hr_double_patchs.shape[3],hr_double_patchs.shape[4])).to(self.device)
            lr_support_x1 = deepcopy(half_support_x1).reshape((-1,3,half_support_x1.shape[2],half_support_x1.shape[3])).to(self.device)

            b, _, h, w = lr_support_x1.shape
            half_condition_weight1=self.Modulation(half_condition_feature1)
            half_condition_weight1 = torch.repeat_interleave(half_condition_weight1, repeats=b//half_condition_weight1.shape[2], dim=2)#[n_block, n_conv_each_block, batch, 1, 64, 1, 1]

            sr_support_x1 = self.forward(lr_support_x1, half_support_x1)
            sr_support_loss = train_pixel_loss(sr_support_x1, hr_support_x1)

            # print("half_condition_feature1:::::::",torch.isnan(half_condition_feature1).any())
            # print("half_condition_feature2:::::::",torch.isnan(half_condition_feature2).any())
            # print("sr_support_x1:::::::", sr_support_x1)
            # print("hr_support_x1:::::::", hr_support_x1)

            condition_loss = contrastive_loss + sr_support_loss*self.coefficient_contrastive_l1

            self.conditionnet_mix_optimizer.zero_grad()
            # contrastive_loss.backward()
            condition_loss.backward()
            self.conditionnet_mix_optimizer.step()
            self.conditionnet_mix_scheduler.step()


            # print("self.Condition.parameters=======", [torch.isnan(i).any() for i in self.Condition.parameters()])
            # print("self.net_g.parameters()=========", [torch.isnan(i).any() for i in self.net_g.parameters()])

            torch.cuda.empty_cache()
        
        tqdm.write("contrastive_loss::::::::::{}".format(contrastive_loss.cpu().detach().numpy()))
        tqdm.write("sr_support_loss::::::::::{}".format(sr_support_loss.cpu().detach().numpy()))
        tqdm.write("condition_loss::::::::::{}".format(condition_loss.cpu().detach().numpy()))
        return condition_loss.cpu().detach().numpy()


    def train_all(self, current_iter):
        lr_double_patchs=self.support_pool
        hr_double_patchs=self.hr
        task_size=lr_double_patchs.shape[0]
        for i in range(2):

            lr_patchs = lr_double_patchs[:,self.support_size*i:self.support_size*(i+1),:,:,:]
            hr_patchs = hr_double_patchs[:,self.support_size*i:self.support_size*(i+1),:,:,:]
            lr_support_patchs = deepcopy(lr_patchs).reshape((task_size,-1,lr_patchs.shape[3],lr_patchs.shape[4])).to(self.device)
            lr_patchs = lr_patchs.reshape((task_size*self.support_size,lr_patchs.shape[2],lr_patchs.shape[3],lr_patchs.shape[4])).to(self.device)
            hr_patchs = hr_patchs.reshape((task_size*self.support_size,hr_patchs.shape[2],hr_patchs.shape[3],hr_patchs.shape[4])).to(self.device)
            
            l1_gt = self.gt_usm
            percep_gt = self.gt_usm
            gan_gt = self.gt_usm

            if self.opt['l1_gt_usm'] is False:
                # l1_gt = self.gt
                l1_gt = hr_patchs

            if self.opt['percep_gt_usm'] is False:
                # percep_gt = self.gt
                percep_gt = hr_patchs
            if self.opt['gan_gt_usm'] is False:
                # gan_gt = self.gt
                gan_gt = hr_patchs

            # optimize net_g
            for p in self.net_d.parameters():
                p.requires_grad = False

            self.optimizer_g.zero_grad()
            # self.output = self.net_g(self.lq)
            # print(self.lq.shape)
            # print(lr_patchs.shape)
            self.output= self.forward(self.lq, lr_support_patchs)

            l_g_total = 0
            loss_dict = OrderedDict()
            if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, l1_gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix
                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style
                # gan loss
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

                # toDo: contrastive loss


                l_g_total.backward()
                self.optimizer_g.step()

            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(gan_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
        '''
        lr_double_patchs=self.support_pool
        hr_double_patchs=self.hr
        task_size=lr_double_patchs.shape[0]
        for i in range(2):
            lr_patchs = lr_double_patchs[:,self.support_size*i:self.support_size*(i+1),:,:,:]
            hr_patchs = hr_double_patchs[:,self.support_size*i:self.support_size*(i+1),:,:,:]
            lr_support_patchs = deepcopy(lr_patchs).reshape((task_size,-1,lr_patchs.shape[3],lr_patchs.shape[4])).to(device)
            lr_patchs = lr_patchs.reshape((task_size*self.support_size,lr_patchs.shape[2],lr_patchs.shape[3],lr_patchs.shape[4])).to(self.device)
            hr_patchs = hr_patchs.reshape((task_size*self.support_size,hr_patchs.shape[2],hr_patchs.shape[3],hr_patchs.shape[4])).to(self.device)

            train_pixel_loss = torch.nn.L1Loss()
            
            self.train_optimizer.zero_grad()
            
            sr_support_patchs = self.forward(lr_patchs, lr_support_patchs)
            sr_l1_loss = train_pixel_loss(sr_support_patchs, hr_patchs)

            sr_l1_loss.backward()
            self.train_optimizer.step()
            self.train_scheduler.step()
            torch.cuda.empty_cache()
        '''
