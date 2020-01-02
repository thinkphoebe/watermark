# -*- coding: utf-8 -*-
"""
@author: Ye Shengnan
create: 2019-12-27
"""
from argparse import ArgumentParser

import cv2
import numpy as np


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--image', dest='img', required=True, help='image file to decode')
    parser.add_argument('--out', dest='out', required=True, help='decoded file to write')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    img = options.img
    out = options.out
    decode(img, out)


def decode(img_path, out_path):
    img = cv2.imread(img_path)
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    cv2.imwrite(out_path, np.real(img_fft))


if __name__ == '__main__':
    main()
