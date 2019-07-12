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

Testing code of SRResNet+ [x2,x3,x4] and SRGAN+ [x4] on Set5.

-- + testsets
   + -- + Set5
        + -- + GT  # ground truth images
        + -- + x2  # low resolution images of scale factor 2
        + -- + x3  # low resolution images of scale factor 3
        + -- + x4  # low resolution images of scale factor 4

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
    utils_logger.logger_info('test_srresnetplus', log_path='test_srresnetplus.log')
    logger = logging.getLogger('test_srresnetplus')

    # basic setting
    # ================================================

    sf = 4  # scale factor
    noise_level_img = 0/255.0  # noise level of L image
    noise_level_model = noise_level_img
    show_img = True

    use_srganplus = True  # 'True' for SRGAN+ (x4) and 'False' for SRResNet+ (x2,x3,x4)
    testsets = 'testsets'
    testset_current = 'Set5'
    n_channels = 3  # only color images, fixed
    border = sf  # shave boader to calculate PSNR and SSIM

    if use_srganplus and sf == 4:
        model_prefix = 'DPSRGAN'
        save_suffix = 'dpsrgan'
    else:
        model_prefix = 'DPSR'
        save_suffix = 'dpsr'

    model_path = os.path.join('DPSR_models', model_prefix+'x%01d.pth' % (sf))

    # --------------------------------
    # L_folder, E_folder, H_folder
    # --------------------------------
    # --1--> L_folder, folder of Low-quality images
    testsubset_current = 'x%01d' % (sf)
    L_folder = os.path.join(testsets, testset_current, testsubset_current)

    # --2--> E_folder, folder of Estimated images
    E_folder = os.path.join(testsets, testset_current, testsubset_current+'_'+save_suffix)
    util.mkdir(E_folder)

    # --3--> H_folder, folder of High-quality images
    H_folder = os.path.join(testsets, testset_current, 'GT')

    need_H = True if os.path.exists(H_folder) else False

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
    logger.info('Model path {:s}. \nTesting...'.format(model_path))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    idx = 0

    logger.info(L_folder)

    for im in os.listdir(os.path.join(L_folder)):
        if im.endswith('.jpg') or im.endswith('.bmp') or im.endswith('.png'):

            logger.info('{:->4d}--> {:>10s}'.format(idx, im)) if not need_H else None

            # --------------------------------
            # (1) img_L
            # --------------------------------
            idx += 1
            img_name, ext = os.path.splitext(im)
            img = util.imread_uint(os.path.join(L_folder, im), n_channels=n_channels)

            np.random.seed(seed=0)  # for reproducibility
            img = util.uint2single(img) + np.random.normal(0, noise_level_img, img.shape)

            util.imshow(img, title='Low-resolution image') if show_img else None

            img_L = util.single2tensor4(img)
            noise_level_map = torch.ones((1, 1, img_L.size(2), img_L.size(3)), dtype=torch.float).mul_(noise_level_model)
            img_L = torch.cat((img_L, noise_level_map), dim=1)
            img_L = img_L.to(device)

            # --------------------------------
            # (2) img_E
            # --------------------------------
            img_E = model(img_L)
            img_E = util.tensor2single(img_E)
            img_E = util.single2uint(img_E)  # np.uint8((z * 255.0).round())

            if need_H:

                # --------------------------------
                # (3) img_H
                # --------------------------------
                img_H = util.imread_uint(os.path.join(H_folder, im), n_channels=n_channels)
                img_H = util.modcrop(img_H, scale=sf)

                # --------------------------------
                # PSNR and SSIM
                # --------------------------------
                psnr = util.calculate_psnr(img_E, img_H, border=border)
                ssim = util.calculate_ssim(img_E, img_H, border=border)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                if np.ndim(img_H) == 3:  # RGB image

                    img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                    img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                    psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
                    ssim_y = util.calculate_ssim(img_E_y, img_H_y, border=border)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)

                    logger.info('{:->20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}.'.format(im, psnr, ssim, psnr_y, ssim_y))
                else:
                    logger.info('{:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(im, psnr, ssim))

            # --------------------------------
            # save results
            # --------------------------------
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None
            util.imsave(img_E, os.path.join(E_folder, img_name+'_x{}'.format(sf)+ext))

    if need_H:

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('PSNR/SSIM(RGB) - {} - x{} -- PSNR: {:.2f} dB; SSIM: {:.4f}'.format(testset_current, sf, ave_psnr, ave_ssim))
        if np.ndim(img_H) == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info('PSNR/SSIM( Y ) - {} - x{} -- PSNR: {:.2f} dB; SSIM: {:.4f}'.format(testset_current, sf, ave_psnr_y, ave_ssim_y))


if __name__ == '__main__':

    main()
