import os.path
import glob
import logging
import cv2

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

Testing code of DPSR [x2,x3,x4] for real image super-resolution.

 -- + testsets
    + -- + real_imgs
         + -- + LR
              + -- + chip.png
              + -- + chip_kernel.png (or chip_kernel.mat --kernel)

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
    utils_logger.logger_info('test_dpsr_real', log_path='test_dpsr_real.log')
    logger = logging.getLogger('test_dpsr_real')

    # basic setting
    # ================================================
    sf = 4
    show_img = True
    noise_level_img = 8./255.
    testsets = 'testsets'
    testset_current = 'real_imgs'

    im = 'chip.png'  # chip.png colour.png

    if 'chip' in im:
        noise_level_img = 8./255.
    elif 'colour' in im:
        noise_level_img = 0.5/255.

    use_srganplus = False
    if use_srganplus and sf == 4:
        model_prefix = 'DPSRGAN'
        save_suffix = 'dpsrgan'
    else:
        model_prefix = 'DPSR'
        save_suffix = 'dpsr'

    model_path = os.path.join('DPSR_models', model_prefix+'x%01d.pth' % (sf))

    iter_num = 15  # number of iterations
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

    # --------------------------------
    # (4) load blur kernel
    # --------------------------------
    if os.path.exists(os.path.join(L_folder, img_name+'_kernel.mat')):
        k = loadmat(os.path.join(L_folder, img_name+'.mat'))['kernel']
        k = k.astype(np.float64)
        k /= k.sum()
    elif os.path.exists(os.path.join(L_folder, img_name+'_kernel.png')):
        k = cv2.imread(os.path.join(L_folder, img_name+'_kernel.png'), 0)
        k = np.float64(k)  # float64 !
        k /= k.sum()
    else:
        k = utils_deblur.fspecial('gaussian', 5, 0.25)
        iter_num = 5

    # --------------------------------
    # (5) handle boundary
    # --------------------------------
    img = utils_deblur.wrap_boundary_liu(img, utils_deblur.opt_fft_size([img.shape[0]+k.shape[0]+1, img.shape[1]+k.shape[1]+1]))

    # --------------------------------
    # (6) get upperleft, denominator
    # --------------------------------
    upperleft, denominator = utils_deblur.get_uperleft_denominator(img, k)

    # --------------------------------
    # (7) get rhos and sigmas
    # --------------------------------
    rhos, sigmas = utils_deblur.get_rho_sigma(sigma=max(0.255/255.0, noise_level_img), iter_num=iter_num)

    # --------------------------------
    # (8) main iteration
    # --------------------------------
    z = img
    rhos = np.float32(rhos)
    sigmas = np.float32(sigmas)

    for i in range(iter_num):

        logger.info('Iter: {:->4d}--> {}'.format(i+1, im))
        # --------------------------------
        # step 1, Eq. (9) // FFT
        # --------------------------------
        rho = rhos[i]
        if i != 0:
            z = util.imresize_np(z, 1/sf, True)

        z = np.real(np.fft.ifft2((upperleft + rho*np.fft.fft2(z, axes=(0, 1)))/(denominator + rho), axes=(0, 1)))

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
    # (9) img_E
    # --------------------------------
    img_E = util.single2uint(z[:h*sf, :w*sf])  # np.uint8((z[:h*sf, :w*sf] * 255.0).round())

    logger.info('saving: sf = {}, {}.'.format(sf, img_name+'_x{}'.format(sf)+ext))
    util.imsave(img_E, os.path.join(E_folder, img_name+'_x{}'.format(sf)+ext))

    util.imshow(img_E, title='Recovered image') if show_img else None


if __name__ == '__main__':

    main()
