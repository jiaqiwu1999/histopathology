import math
import os
import numpy as np
import openslide
import shutil
import json
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
from openslide import OpenSlideError
import pandas as pd

def get_tile_from_original_slide(slide, patch_size, target_mag, save_dir, patient_id):
    print("Processing tile extraction for patient ", patient_id)
    if ('aperio.AppMag' not in slide.properties):
        return
    magnification = float(slide.properties['aperio.AppMag'])
    extract_patch_size = int(patch_size * magnification / target_mag)
    w, h = slide.level_dimensions[0]
    w = w // extract_patch_size * extract_patch_size
    h = h // extract_patch_size * extract_patch_size
    count = 0
    num_patch = 0
    if (save_dir/patient_id).exists():
        return
        shutil.rmtree(save_dir/patient_id)
    (save_dir/patient_id).mkdir()
    for i in range(0, w, extract_patch_size):
        for j in range(0, h, extract_patch_size):
            patch = slide.read_region((i, j), level=0, size=[extract_patch_size, extract_patch_size])
            patch = patch.resize([patch_size, patch_size])
            patch_gray = patch.convert('1')
            ave_pixel_val = np.array(patch_gray).mean()
            if ave_pixel_val < threshold:
                num_patch += 1
                img_name = patient_id + '_' + str(count).zfill(4) + '.png'
                tile_name = save_dir/patient_id/img_name
                patch.save(str(tile_name))
            count += 1
    # print(f"Number of tissue patch is {num_patch}")
    # print(f"Number of total patches is {count}")
    print("Finished processing -------")
    
# some globals, can be changed to customize
target_mag = 20
threshold = 200/255
patch_size = 512
data_dir = Path('../SVS_Raw/')
folders = [x for x in data_dir.iterdir() if x.is_dir()]
current_slides = []
current_patients = []

if (Path('../all_patients_id.txt').exists()):
    print("Found existing patients, loading it...")
    with open('../all_patients_id.txt', 'r') as fread:
        existing_patients = fread.read().split('\n')

    
for folder in folders:
    current_slide = list(folder.glob('*.svs'))[0]
    slide_name = str(current_slide)
    # print(slide_name)
    current_patient = slide_name.split('/')[-1][:15]
    # only select 1 slide from each patient
    if current_patient in existing_patients:
        continue
    current_patients.append(current_patient)
    current_slides.append(slide_name)
    

# with open('../all_patients_id.txt', 'w') as fwrite:
#     for p in current_patients:
#         fwrite.write(p)
#         fwrite.write('\n')
        
save_dir_root = Path('../tile_data/')
    
def get_thumbnail(save_dir, slide, patient_id, target_mg=20):

    magnification = float(slide.properties['aperio.AppMag'])
    # print(magnification)

    extract_patch_size = int(patch_size * magnification / target_mag)
    # print(extract_patch_size)
    w, h = slide.level_dimensions[0]

    th_w = int(w / extract_patch_size * 10)
    th_h = int(h / extract_patch_size * 10)
    thumbnail = slide.get_thumbnail((th_w, th_h))
#     thumbnail_name = '{:s}/{:s}_thumbnail.png'.format(save_dir, slide_name)
    save_path = os.path.join(save_dir, f"thumbnail_{patient_id}.png")
    thumbnail.save(save_path)
    
slide_info = {}
with open('../slide_info.json', 'r') as fread:
    slide_info = json.load(fread)
    
thumb_save_dir = '../thumbnail_data'
finished = []
additional = []
for i in range(len(current_slides)):
    if (current_patients[i] not in existing_patients):
        slide = openslide.open_slide(str(current_slides[i]))
        w, h = slide.level_dimensions[0]
        slide_info[current_patients[i]] = {'mag': float(slide.properties['aperio.AppMag']), 'w': w, 'h': h}
        get_thumbnail(thumb_save_dir, slide, current_patients[i])
        print("Adding tiles")
        get_tile_from_original_slide(slide, patch_size, target_mag, save_dir_root, current_patients[i])
        additional.append(current_patients[i])
    # if (current_patients[i]) in existing_patients:
    #     # check slide stats
    #     slide = openslide.open_slide(str(current_slides[i]))
    #     w, h = slide.level_dimensions[0]
    #     if (w == slide_info[current_patients[i]]['w'] and h == slide_info[current_patients[i]]['h']):
    #         # get_thumbnail(thumb_save_dir, slide, current_patients[i])
    #         print("Re-tiling...")
    #         get_tile_from_original_slide(slide, patch_size, target_mag, save_dir_root, current_patients[i])
    #         finished.append(current_patients[i])
            
    

addition =True
print("In addition mode... Computing new additional subject lists")
if (addition):
    with open('../additional.txt', 'w') as fr:
        for patient in current_patients:
            if patient not in existing_patients:
                fr.write(patient)
                fr.write('\n')
    with open('../slide_info.json', 'w') as fwrite:
        json.dump(slide_info, fwrite, indent=4)
# info = {}

# if Path('../slide_info.json').exists():
#     with open('../slide_info.json', 'r') as f:
#         info = json.load(f)

# for i in range(len(current_slides)):
#     if current_patients[i] not in info:
#         info[current_patients[i]] = {}
#         slide = openslide.open_slide(str(current_slides[i]))
#         magnification = float(slide.properties['aperio.AppMag'])
#         w, h = slide.level_dimensions[0]
#         info[current_patients[i]]['mag'] = magnification
#         info[current_patients[i]]['w'] = w
#         info[current_patients[i]]['h'] = h
    
