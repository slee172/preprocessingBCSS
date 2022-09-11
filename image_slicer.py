import os
import numpy as np
import collections
import concurrent.futures
from PIL import Image
from tqdm import tqdm
from itertools import repeat
from color_conversion import lab_mean_std
from reinhard import reinhard
Image.MAX_IMAGE_PIXELS = None  # to avoid image size warning


def parallel_cropping(i, mesh_x, mesh_y, img_size, i_dir, m_dir, img_out, mask_out, i_name):
    i_path = i_dir + i_name
    m_path = m_dir + i_name
    # begin color normalization
    # generate a sampling of sample_pixels_rgb pixels from whole-slide image
    i_rgb = Image.open(i_path).convert('RGB')
    a_rgb = np.array(i_rgb)
    # reshape the Nx3 pixel array into a 1 x N x 3 image for lab_mean_std
    pixels_rgb = np.reshape(a_rgb,
                            (1, a_rgb.shape[0] * a_rgb.shape[1], 3))
    # compute mean and stddev of sample pixels in Lab space
    mu, sigma = lab_mean_std(pixels_rgb)
    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    src_mu_lab, src_sigma_lab = ReinhardStats(mu, sigma)
    img_norm = reinhard(a_rgb, reference_mu_lab,
                        reference_std_lab, src_mu=src_mu_lab,
                        src_sigma=src_sigma_lab)
    img_norm = Image.fromarray(img_norm)
    # end color normalization
    mask = Image.open(m_path)
    for j in range(X.shape[1] - 1):
        crop_name = i_name.split('_')[0] + '_' + str(mesh_x[i, j]) + '_' + str(mesh_y[i, j]) + '_' + str(
            img_size) + '_' + str(img_size) + '_' + '.png'
        crop_img = img_norm.crop((mesh_y[i, j], mesh_x[i, j], mesh_y[i, j] + img_size, mesh_x[i, j] + img_size))
        crop_mask = mask.crop((mesh_y[i, j], mesh_x[i, j], mesh_y[i, j] + img_size, mesh_x[i, j] + img_size))
        crop_img.save(img_out + crop_name)
        crop_mask.save(mask_out + crop_name)


if __name__ == '__main__':
    # img_dir = '/home/leesan/workspace/CrowdsourcingDataset-Amgadetal2019/images/'
    # mask_dir = '/home/leesan/workspace/CrowdsourcingDataset-Amgadetal2019/masks/'
    img_dir = 'your img directory'
    mask_dir = 'your mask directory'
    img_list = sorted(os.listdir(img_dir))
    mask_list = sorted(os.listdir(mask_dir))
    # img_dir_out = '/home/leesan/workspace/CrowdsourcingDataset-Amgadetal2019/sliced/images/'
    # mask_dir_out = '/home/leesan/workspace/CrowdsourcingDataset-Amgadetal2019/sliced/annots/'
    img_dir_out = 'your image output directory'
    mask_dir_out = 'your mask output directory'
    stride = 256
    # we use reinhard color normalization
    global reference_mu_lab, reference_std_lab
    reference_mu_lab = [8.63234435, -0.11501964, 0.03868433]
    reference_std_lab = [0.57506023, 0.10403329, 0.01364062]

    for img_name in tqdm(img_list):
        img_path = img_dir + img_name
        img_rgb = Image.open(img_path).convert('RGB')
        array_rgb = np.asarray(img_rgb)
        height, width, channel = array_rgb.shape
        X = np.arange(0, width + 1, stride)
        Y = np.arange(0, height + 1, stride)
        X, Y = np.meshgrid(X, Y)
        # parallel-running
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            for _ in executor.map(parallel_cropping, list(range(X.shape[0] - 1)), repeat(X), repeat(Y), repeat(stride),
                                  repeat(img_dir), repeat(mask_dir), repeat(img_dir_out), repeat(mask_dir_out),
                                  repeat(img_name)):
                pass
