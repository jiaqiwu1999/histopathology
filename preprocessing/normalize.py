import os
import stainNorm_Reinhard
from skimage import color, io
import argparse
import numpy as np
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import shutil
import json
import matplotlib.pyplot as plt

def compute_mean_std(data_path, file_list, ratio=1.0):
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels

    random.shuffle(file_list)
    N = len(file_list)
    files_to_be_processed = file_list[:int(ratio*N)]

    for file_name in files_to_be_processed:
        img = io.imread(file_name)
        img = img[:, :, :3]

        img = color.rgb2lab(img)

        total_sum += img.sum(axis=(0, 1))
        total_square_sum += (img ** 2).sum(axis=(0, 1))
        num_pixel += img.shape[0] * img.shape[1]

    # compute the mean values of each channel
    mean_values = total_sum / num_pixel
    # compute the standard deviation
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)

    return mean_values, std_values


def cluster_to_file(cluster, patient_id):
    files = []
    for num in cluster:
        img_name = f'../tile_data/{patient_id}/'+ patient_id + '_' + str(num).zfill(4) + '.png'
        files.append(img_name)
    return files


ref_slide = 'TCGA-XF-A9T3-01'
ref_img_list = list(glob.glob(f'../tile_data/{ref_slide}/*.png'))

means, stds = compute_mean_std('../tile_data/{:s}/'.format(ref_slide, ref_slide), ref_img_list, ratio=0.2)


normalizer = stainNorm_Reinhard.normalizer(means, stds)

with open('../all_patients_id.txt', 'r') as fread:
    patients = fread.read().splitlines()
    
save_folder = Path('../normalized_data')
if not save_folder.exists():
    save_folder.mkdir()

normalize_info = {}
if Path('../normalized.json').exists():
    with open('../normalized.json', 'r') as fr:
        normalize_info = json.load(fr)

    
for patient in patients:
    print(f'--Normalizing for patient {patient}---')
    cluster_file = f'../cluster_data/{patient}_clustering_results.npy'
    cluster = np.load(cluster_file, allow_pickle=True)
    tumor_cluster = cluster.item().get('cluster_high')
    tumor_img_list = cluster_to_file(tumor_cluster, patient)
    
    dest = save_folder/patient
    
    if dest.exists():
        continue
    
    dest.mkdir()
    slide_mean, slide_std = compute_mean_std('../tile_data/{:s}/'.format(patient), tumor_img_list, ratio=0.2)
    
    normalize_info[patient] = {'means': means, 'stds': stds}
    for img_file in tumor_img_list:
        img = io.imread(img_file)
        img = img[:, :, :3]

        # perform reinhard color normalization
        img_normalized = normalizer.transform(img, slide_mean, slide_std)
        
        save_name = img_file.split('/')[-1]
        io.imsave(dest/save_name, img_normalized)
    print('---Finished Normalizing---')
    
    
with open('../normalized.json', 'w') as fwrite:
    json.dump(normalize_info, fwrite, indent=4)