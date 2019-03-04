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

Testing code of SRResNet+ [x2,x3,x4] and SRGAN+ [x4] for real image super-resolution.

 -- + testsets
    + -- + real_imgs
         + -- + LR
              + -- + frog.png

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
    utils_logger.logger_info('test_srresnetplus_real', log_path='test_srresnetplus_real.log')
    logger = logging.getLogger('test_srresnetplus_real')

    # basic setting
    # ================================================

    sf = 4  # from 2, 3 and 4
    noise_level_img = 14./255.  # noise level of low-quality image
    testsets = 'testsets'
    testset_current = 'real_imgs'
    use_srganplus = True  # 'True' for SRGAN+ (x4) and 'False' for SRResNet+ (x2,x3,x4)

    im = 'frog.png'  # frog.png

    if 'frog' in im:
        noise_level_img = 14./255.

    noise_level_model = noise_level_img  # noise level of model

    if use_srganplus and sf == 4:
        model_prefix = 'DPSRGAN'
        save_suffix = 'srganplus'
    else:
        model_prefix = 'DPSR'
        save_suffix = 'srresnet'

    model_path = os.path.join('DPSR_models', model_prefix+'x%01d.pth' % (sf))
    show_img = True
    n_channels = 3  # only color images, fixed

    # ================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # (1) load trained model
    # --------------------------------

    model = SRResNet(in_nc=4, out_nc=3, nc=96, nb=16, upscale=sf, act_mode='R', upsample_mode='pixelshuffle')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path {:s}. Testing...'.format(model_path))

    # --------------------------------
    # (2) L_folder, E_folder
    # --------------------------------
    # --1--> L_folder, folder of Low-quality images
    L_folder = os.path.join(testsets, testset_current, 'LR')  # L: Low quality

    # --2--> E_folder, folder of Estimated images
    E_folder = os.path.join(testsets, testset_current, 'x{:01d}_'.format(sf)+save_suffix)
    util.mkdir(E_folder)

    logger.info(L_folder)

    # for im in os.listdir(os.path.join(L_folder)):
    #   if (im.endswith('.jpg') or im.endswith('.bmp') or im.endswith('.png')) and 'kernel' not in im:

    # --------------------------------
    # (3) load low-resolution image
    # --------------------------------
    img_name, ext = os.path.splitext(im)
    img = util.imread_uint(os.path.join(L_folder, im), n_channels=n_channels)
    h, w = img.shape[:2]
    util.imshow(img, title='Low-resolution image') if show_img else None
    img = util.unit2single(img)
    img_L = util.single2tensor4(img)

    # --------------------------------
    # (4) do super-resolution
    # --------------------------------
    noise_level_map = torch.ones((1, 1, img_L.size(2), img_L.size(3)), dtype=torch.float).mul_(noise_level_model)
    img_L = torch.cat((img_L, noise_level_map), dim=1)
    img_L = img_L.to(device)
    # with torch.no_grad():
    img_E = model(img_L)
    img_E = util.tensor2single(img_E)

    # --------------------------------
    # (5) img_E
    # --------------------------------
    img_E = util.single2uint(img_E[:h*sf, :w*sf])  # np.uint8((z[:h*sf, :w*sf] * 255.0).round())

    logger.info('saving: sf = {}, {}.'.format(sf, img_name+'_x{}'.format(sf)+ext))
    util.imsave(img_E, os.path.join(E_folder, img_name+'_x{}'.format(sf)+ext))

    util.imshow(img_E, title='Recovered image') if show_img else None


if __name__ == '__main__':

    main()
