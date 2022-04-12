import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class tileDataset(Dataset):
    def __init__(self, tile_root_directory, patient_list, label_dict, data_transform, N):
        super(tileDataset, self).__init__()
        self.label_dict = label_dict # a Path object
        self.root_dir = tile_root_directory
        self.patients = patient_list
        self.transform = data_transform
        self.N = N
        
    def __getitem__(self, idx):
        curr_patient = self.patients[idx]
        # get tiles for this patient
        lookup_dir = self.root_dir/curr_patient
        tile_names = [x for x in lookup_dir.iterdir() if x.is_file()]
        if len(tile_names) > self.N:
            print(f"Input for patient {curr_patient} exceeds limit, truncating..")
            tile_names = tile_names[:self.N]
        output = []
        for t in tile_names:
            img = Image.open(t).convert('RGB')
            img = self.transform(img).unsqueeze(0)
            output.append(img)
        output = torch.cat(output, dim=0)
        label = float(self.label_dict[curr_patient])
        return torch.permute(output, [1, 0, 2, 3]), torch.tensor(label)
            
            
    def __len__(self):
        return len(self.patients)
            
        
class tileFeatureDataset(Dataset):
    def __init__(self, feature_files, label_dict):
        super(tileFeatureDataset, self).__init__()
        self.feature_files = feature_files
        self.label_dict = label_dict
        
    def __getitem__(self, idx):
        f = self.feature_files[idx]
        feature = np.load(f)
        patient = f.split('/')[-1].split('.')[0]
        label = float(label_dict[patient])
        return torch.from_numpy(feature), torch.tensor(label)
        
    def __len__(self):
        return len(self.feature_files)