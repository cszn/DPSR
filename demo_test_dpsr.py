import os.path
import glob
import cv2
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict
from scipy.io import loadmat

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_image as util
from models.network_srresnet import SRResNet


'''
Spyder (Python 3.6)
PyTorch 0.4.1
Windows 10

Testing code of DPSR [x2,x3,x4] (and DPSRGAN [x4]) on
BSD68 with SRResNet+ [x2,x3,x4] (and  SRGAN+ [x4]).

Three types of blur kernels, i.e.,
(g) Gaussian blur kernels,
(m) motion blur kernels, and
(d) disk blur kernels,
are considered.

 -- + testsets
    + -- + BSD68
         + -- + GT    # ground truth images
         + -- + x2_d  # low-resolution images of scale factor 2 with disk blur kernels
         + -- + x3_d
         + -- + x4_d
         + -- + x2_g
         + -- + x3_g  # low-resolution images of scale factor 3 with Gaussian blur kernels
         + -- + x4_g
         + -- + x2_m
         + -- + x3_m
         + -- + x4_m  # low-resolution images of scale factor 4 with motion blur kernels

You can generate x2_d, ..., x4_m by generate_blurry_LR_images.m with Matlab or
you can download x2_d, ..., x4_m from:
https://drive.google.com/file/d/1IThQ0kZGL71pfIry5qzCoW0DftqLleOC/view?usp=sharing

For more information, please refer to the following paper.

@inproceedings{zhang2019deep,
  title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2019}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (03/03/2019)
'''


def main():

    # --------------------------------
    # let's start!
    # --------------------------------
    utils_logger.logger_info('test_dpsr', log_path='test_dpsr.log')
    logger = logging.getLogger('test_dpsr')

    # basic setting
    # ================================================

    sf = 4  # scale factor
    noise_level_img = 0/255.0  # noise level of low quality image, default 0
    noise_level_model = noise_level_img  # noise level of model, default 0
    show_img = True

    use_srganplus = True  # 'True' for SRGAN+ (x4) and 'False' for SRResNet+ (x2,x3,x4)
    testsets = 'testsets'
    testset_current = 'BSD68'

    if use_srganplus and sf == 4:
        model_prefix = 'DPSRGAN'
        save_suffix = 'dpsrgan'
    else:
        model_prefix = 'DPSR'
        save_suffix = 'dpsr'

    model_path = os.path.join('DPSR_models', model_prefix+'x%01d.pth' % (sf))

    iter_num = 15  # number of iterations, fixed
    n_channels = 3  # only color images, fixed
    border = sf**2  # shave boader to calculate PSNR, fixed

    # k_type = ('d', 'm', 'g')
    k_type = ('m')  # motion blur kernel

    # ================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # load model
    # --------------------------------
    model = SRResNet(in_nc=4, out_nc=3, nc=96, nb=16, upscale=sf, act_mode='R', upsample_mode='pixelshuffle')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path {:s}. Testing...'.format(model_path))

    # --------------------------------
    # read image (img) and kernel (k)
    # --------------------------------
    test_results = OrderedDict()

    for k_type_n in range(len(k_type)):

        # --1--> L_folder, folder of Low-quality images
        testsubset_current = 'x%01d_%01s' % (sf, k_type[k_type_n])
        L_folder = os.path.join(testsets, testset_current, testsubset_current)

        # --2--> E_folder, folder of Estimated images 
        E_folder = os.path.join(testsets, testset_current, testsubset_current+'_'+save_suffix)
        util.mkdir(E_folder)

        # --3--> H_folder, folder of High-quality images
        H_folder = os.path.join(testsets, testset_current, 'GT')

        test_results['psnr_'+k_type[k_type_n]] = []

        logger.info(L_folder)
        idx = 0

        for im in os.listdir(os.path.join(L_folder)):
            if im.endswith('.jpg') or im.endswith('.bmp') or im.endswith('.png'):

                # --------------------------------
                # (1) img_L
                # --------------------------------
                idx += 1
                img_name, ext = os.path.splitext(im)
                img_L = util.imread_uint(os.path.join(L_folder, im), n_channels=n_channels)
                util.imshow(img_L) if show_img else None

                np.random.seed(seed=0)  # for reproducibility
                img_L = util.uint2single(img_L) + np.random.normal(0, noise_level_img, img_L.shape)

                # --------------------------------
                # (2) kernel
                # --------------------------------
                k = loadmat(os.path.join(L_folder, img_name+'.mat'))['kernel']
                k = k.astype(np.float32)
                k /= np.sum(k)

                # --------------------------------
                # (3) get upperleft, denominator
                # --------------------------------
                upperleft, denominator = utils_deblur.get_uperleft_denominator(img_L, k)

                # --------------------------------
                # (4) get rhos and sigmas
                # --------------------------------
                rhos, sigmas = utils_deblur.get_rho_sigma(sigma=max(0.255/255., noise_level_model), iter_num=iter_num)

                # --------------------------------
                # (5) main iteration
                # --------------------------------
                z = img_L
                rhos = np.float32(rhos)
                sigmas = np.float32(sigmas)

                for i in range(iter_num):

                    # --------------------------------
                    # step 1, Eq. (9) // FFT
                    # --------------------------------
                    rho = rhos[i]
                    if i != 0:
                        z = util.imresize_np(z, 1/sf, True)

                    z = np.real(np.fft.ifft2((upperleft + rho*np.fft.fft2(z, axes=(0, 1)))/(denominator + rho), axes=(0, 1)))
                    # imsave('LR_deblurred_%02d.png'%i, np.clip(z, 0, 1))

                    # --------------------------------
                    # step 2, Eq. (12) // super-resolver
                    # --------------------------------
                    sigma = torch.from_numpy(np.array(sigmas[i]))
                    img_L = util.single2tensor4(z)

                    noise_level_map = torch.ones((1, 1, img_L.size(2), img_L.size(3)), dtype=torch.float).mul_(sigma)
                    img_L = torch.cat((img_L, noise_level_map), dim=1)
                    img_L = img_L.to(device)
                    # with torch.no_grad():
                    z = model(img_L)
                    z = util.tensor2single(z)

                # --------------------------------
                # (6) img_E
                # --------------------------------
                img_E = util.single2uint(z)  # np.uint8((z * 255.0).round())

                # --------------------------------
                # (7) img_H
                # --------------------------------
                img_H = util.imread_uint(os.path.join(H_folder, img_name[:7]+'.png'), n_channels=n_channels)

                util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

                psnr = util.calculate_psnr(img_E, img_H, border=border)
                
                logger.info('{:->4d}--> {:>10s}, {:.2f}dB'.format(idx, im, psnr))
                test_results['psnr_'+k_type[k_type_n]].append(psnr)

                util.imsave(img_E, os.path.join(E_folder, img_name+ext))

        ave_psnr = sum(test_results['psnr_'+k_type[k_type_n]]) / len(test_results['psnr_'+k_type[k_type_n]])
        logger.info('------> Average PSNR(RGB) of ({} - {}) is : {:.2f} dB'.format(testset_current, testsubset_current, ave_psnr))


if __name__ == '__main__':

    main()
