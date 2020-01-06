# -*- coding: utf-8 -*-
"""
@author: Ye Shengnan
create: 2019-12-27
"""
import sys
from argparse import ArgumentParser

import cv2
import numpy as np


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--image', dest='img', required=True, help='image file to add watermark')
    parser.add_argument('--watermark', dest='wm', required=True, help='watermark image file')
    parser.add_argument('--out', dest='out', required=True, help='output image file')
    parser.add_argument('--alpha', dest='alpha', default=10)
    parser.add_argument('--size_percent', dest='size_percent', default=30, help='watermark size ratio to source')
    parser.add_argument('--offset_percent', dest='offset_percent', default=5, help='watermark offset to source center')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    img = options.img
    wm = options.wm
    out = options.out
    alpha = float(options.alpha)
    size_percent = float(options.size_percent)
    offset_percent = float(options.offset_percent)
    if size_percent + offset_percent > 50:
        print('size_percent + offset_percent should less than 50')
        return sys.exit(-1)
    encode(img, wm, out, alpha, size_percent, offset_percent)


def encode(img_path, wm_path, out_path, alpha, size_percent, offset_percent):
    # read source
    img = cv2.imread(img_path)
    img_h, img_w, img_c = np.shape(img)
    print('image width:%d, height:%d, channels:%d' % (img_w, img_h, img_c))

    # read watermark
    wm = cv2.imread(wm_path, cv2.IMREAD_UNCHANGED)
    wm_h, wm_w, wm_c = np.shape(wm)
    print('watermark width:%d, height:%d, channels:%d' % (wm_w, wm_h, wm_c))

    # merge watermark alpha
    if wm_c == 4:
        alpha_channel = wm[:, :, 3]
        rgb_channels = wm[:, :, :3]
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)
        wm = rgb_channels.astype(np.float32) * alpha_factor

    # scale watermark to percent size
    dst_w = int(img_w * size_percent / 100)
    dst_h = int(img_h * size_percent / 100)
    scale = dst_w / wm_w
    if dst_w / wm_w > dst_h / wm_h:
        scale = dst_h / wm_h
    dst_w = int(wm_w * scale)
    dst_h = int(wm_h * scale)
    wm_scaled = cv2.resize(wm, dsize=(dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
    scaled_h, scaled_w, scaled_c = np.shape(wm_scaled)
    print('dst_w:%d, dst_h:%d, scaled_w:%d, scaled_h:%d, scaled_c:%d' % (dst_w, dst_h, scaled_w, scaled_h, scaled_c))

    # place watermark to transparent image
    offset_x = int(img_w * offset_percent / 100)
    offset_y = int(img_h * offset_percent / 100)
    print('offset_x:%d, offset_y:%d' % (offset_x, offset_y))
    offset_x = int(img_w / 2) - offset_x - scaled_w
    offset_y = int(img_h / 2) - offset_y - scaled_h
    print('modified offset_x:%d, offset_y:%d' % (offset_x, offset_y))
    tmp = np.zeros(img.shape)
    for i in range(scaled_h):
        for j in range(scaled_w):
            tmp[i + offset_y][j + offset_x] = wm_scaled[i][j]
            # tmp[i][width - 1 - j] = watermark[i][j]

    # fft
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    # overlap
    img_fft_overlap = img_fft + alpha * tmp

    # ifft
    img_fft_overlap = np.fft.ifftshift(img_fft_overlap)
    img_out = np.fft.ifft2(img_fft_overlap)

    cv2.imwrite(out_path, np.real(img_out))


if __name__ == '__main__':
    main()
