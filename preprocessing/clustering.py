import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import measure, color, io
import openslide
from PIL import Image
from sklearn.cluster import KMeans
# morphology operation
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu # seperate background & foreground
from skimage.morphology import closing, opening, disk
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.segmentation import clear_border



def waterShed(file_name, kernel_size=3, perc=0.2):
    im = cv2.imread(file_name)
    cells = im[:, :, 0]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    cells = hsv[:, :, 1]
    ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = clear_border(opening)
    #plt.imshow(opening, cmap='gray')
    sure_bg = cv2.dilate(opening, kernel, iterations=5)
    plt.imshow(sure_bg, cmap='gray')
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    return dist_transform.max()

save_dir = Path('../cluster_data/')
if not save_dir.exists():
    save_dir.mkdir()
    
    
def cluster_one_patient_2_cluster(patient_id, tiles, tiles_index, features):
    print("Performing clustering for patient ", patient_id)
    stats = []
    flattened = []
    for i in range(len(tiles)):
        file = tiles[i]
        dist = waterShed(str(file))
        image = Image.open(file)
        image2 = image.resize((128, 128))   # for 512x512 patch
        image2 = np.array(image2)[:, :, :3]
        flattened.append(image2.flatten())
        stats.append(dist)
    stats = np.asarray(stats)
    flattened = np.asarray(flattened)
    high_perc = np.percentile(stats, 75)
    low_perc = np.percentile(stats, 25)
    cluster_high = tiles_index[stats >= high_perc]
    cluster_low = tiles_index[stats < high_perc]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(flattened)
    cluster1 = tiles_index[kmeans.labels_ == 0]
    cluster2 = tiles_index[kmeans.labels_ == 1]
    clustering_results = {'cluster_high': cluster_high,
                          'cluster_low': cluster_low,
                          'knn_img_c1': cluster1,
                          'knn_img_c2': cluster2
                          }
    save_name = f'{patient_id}_clustering_results.npy'
    np.save(save_dir/save_name, clustering_results)
    print("Finised clustering for patient ", patient_id)
    
    
# def cluster_one_patient_3_cluster(patient_id, tiles, tile_index, features):
#     stats = []
#     flattened = []
#     for i in range(len(tiles)):
#         file = tiles[i]
#         dist = waterShed(str(file))
#         image = Image.open(file)
#         image2 = image.resize((128, 128))   # for 512x512 patch
#         image2 = np.array(image2)[:, :, :3]
#         flattened.append(image2.flatten())
#         stats.append(dist)
#     stats = np.asarray(stats)
#     flattened = np.asarray(flattened)
#     high_perc = np.percentile(stats, 75)
#     low_perc = np.percentile(stats, 25)
#     cluster_high = tile_index[stats >= high_perc]
#     cluster_mid = tile_index[(stats < high_perc) & (stats > low_perc)]
#     cluster_low = tile_index[stats <= low_perc]
#     kmeans = KMeans(n_clusters=3, random_state=0).fit(flattened)
#     cluster1 = tile_index[kmeans.labels_ == 0]
#     cluster2 = tiles_index[kmeans.labels_ == 1]
#     cluster3 = tiles_index[kmeans.labels_ == 2]
#     # kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
#     # cluster4 = tile_index[kmeans.labels_ == 0]
#     # cluster5 = tile_index[kmeans.labels_ == 1]
#     # cluster6 = tile_index[kmeans.labels_ == 2]
#     clustering_results = {'cluster_high': cluster_high,
#                           'cluster_mid': cluster_mid,
#                           'cluster_low': cluster_low,
#                           'knn_img_c1': cluster1,
#                           'knn_img_c2': cluster2,
#                           'knn_img_c3': cluster3,
#                           }
#     np.save(f'{patient_id}_clustering_results.npy', clustering_results)
    
    
# patient_id_file = Path('../all_patients_id.txt')
patient_id_file = Path('../additional.txt')
patient_id = []
with open(patient_id_file, 'r') as f:
    patient_id = f.read().splitlines()

patient_id = np.unique(patient_id)
tile_root = Path('../tile_data')
feature_root = Path('../feature_data')
cluster_root = Path('../cluster_data')
for patient in patient_id:
    save_name = f'{patient}_clustering_results.npy'
    patient_dir = tile_root/patient
    if (cluster_root/save_name).exists():
        continue
    tiles = [x for x in patient_dir.iterdir() if x.is_file()]
    tiles.sort()
    tile_index = []
    for i in range(len(tiles)):
        num = int(str(tiles[i]).split('/')[-1].split('.')[0].split('_')[-1])
        tile_index.append(num)
    tile_index = np.asarray(tile_index)
    features = None
    cluster_one_patient_2_cluster(patient, tiles, tile_index, features)