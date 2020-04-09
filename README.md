# DPSR

# Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels (CVPR, 2019)


# Training and testing codes for the super-resolver prior ([PyTorch](https://github.com/cszn/KAIR))
- [main_train_dpsr.py](https://github.com/cszn/KAIR/blob/master/main_train_dpsr.py)

- [main_test_dpsr.py](https://github.com/cszn/KAIR/blob/master/main_test_dpsr.py)

***

The left is the blurry LR image. The right is the super-resolved image by DPSRGAN with scale factor 4.

Run [demo_test_dpsr.py](demo_test_dpsr.py) to produce the following results.


<img src="testsets/BSD68/x4_m/test_02_m_13.png" width="72px"/> <img src="testsets/BSD68/x4_m_dpsrgan/test_02_m_13.png" width="288px"/>
<img src="testsets/BSD68/x4_m/test_03_m_24.png" width="72px"/> <img src="testsets/BSD68/x4_m_dpsrgan/test_03_m_24.png" width="288px"/>

<img src="testsets/BSD68/x4_m/test_39_m_32.png" width="72px"/> <img src="testsets/BSD68/x4_m_dpsrgan/test_39_m_32.png" width="288px"/>
<img src="testsets/BSD68/x4_m/test_01_m_19.png" width="72px"/> <img src="testsets/BSD68/x4_m_dpsrgan/test_01_m_19.png" width="288px"/>

<img src="testsets/BSD68/x4_m/test_14_m_14.png" width="120px"/> <img src="testsets/BSD68/x4_m_dpsrgan/test_14_m_14.png" width="480px"/>

<img src="testsets/BSD68/x4_m/test_33_m_06.png" width="120px"/> <img src="testsets/BSD68/x4_m_dpsrgan/test_33_m_06.png" width="480px"/>

***

Super-resolved images of LR image [chip.png](testsets/real_imgs/LR/chip.png) by DPSR with scale factors 2, 3 and 4.

Run [demo_test_dpsr_real.py](demo_test_dpsr_real.py) to produce the following results.


<img src="testsets/real_imgs/LR/chip.png" width="109px"/>  LR

<img src="testsets/real_imgs/x4_dpsr/chip_x2.png" width="218px"/>  x2

<img src="testsets/real_imgs/x4_dpsr/chip_x3.png" width="327px"/>  x3

<img src="testsets/real_imgs/x4_dpsr/chip_x4.png" width="436px"/>  x4






# Requirements and Dependencies
- Spyder (Python 3.6)
- PyTorch 0.4.1
- Windows 10


# Citation
```
@inproceedings{zhang2019deep,
  title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1671--1681},
  year={2019}
}
```
