import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import os

def generate_density_map(image_shape, locations, sigma=15):
    density_map = np.zeros(image_shape[:2], dtype=np.float32)
    for point in locations:
        x = min(int(point[0]), image_shape[1] - 1)
        y = min(int(point[1]), image_shape[0] - 1)
        density_map[y, x] = 1
    return gaussian_filter(density_map, sigma=sigma)

image_dir = r'archive\ShanghaiTech\part_B\test_data\images'
gt_dir = r'archive\ShanghaiTech\part_B\test_data\ground-truth'
output_dir = r'archive\ShanghaiTech\part_B\test_data\density-map'

for gt_filename in os.listdir(gt_dir):
    if not gt_filename.endswith('.mat'):
        continue

    iamge_filename = gt_filename.replace("GT_", '').replace('.mat', '.jpg')
    image_path = os.path.join(image_dir, iamge_filename)
    gt_path = os.path.join(gt_dir, gt_filename)

    image = cv2.imread(image_path)
    if image is None:
        continue

    mat = scipy.io.loadmat(gt_path)

    locations = mat['image_info'][0][0][0][0][0]
    person_count = mat['image_info'][0][0][0][0][1][0][0]

    density_map = generate_density_map(image.shape, locations)
    
    output_path = os.path.join(output_dir, iamge_filename.replace('.jpg', '.npy'))
    np.save(output_path, density_map)
    estimated = density_map.sum()

    print(f'{iamge_filename} | GT : {person_count} | Density sum : {estimated:.2f}')
