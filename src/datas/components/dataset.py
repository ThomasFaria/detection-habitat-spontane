import random
from typing import List, Optional, Union

import numpy as np
import torch
from albumentations import Compose
from torch.utils.data import Dataset
from satellite_image import SatelliteImage
from labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)
from change_detection_triplet import ChangedetectionTripletS2Looking
    

# TODO: pour le moment que pour la segmentation
class SatelliteDataset(Dataset):
    """
    Custom Dataset class.
    """

    def __init__(
        self,
        labeled_images: Union[
            List[SegmentationLabeledSatelliteImage],
            List[DetectionLabeledSatelliteImage],
        ],
        transforms: Optional[Compose] = None,
        bands_indices: Optional[List[int]] = None,
    ):
        """
        Constructor.

        Args:
            labeled_images (List): _description_
            transforms (Optional[Compose]): Compose object from albumentations.
            bands_indices (List): List of indices of bands to plot.
                The indices should be integers between 0 and the
                number of bands - 1.
        """
        self.labeled_images = labeled_images
        self.transforms = transforms
        self.bands_indices = bands_indices

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labeled_image = self.labeled_images[idx]
        satellite_image = labeled_image.satellite_image.array[
            self.bands_indices, :, :
        ].squeeze()
        mask = labeled_image.label

        sample = {"image": satellite_image, "mask": mask}
        if self.transforms:
            satellite_image = np.transpose(satellite_image, (1, 2, 0))
            sample = self.transforms(image=satellite_image, mask=mask)
        satellite_image = sample["image"]
        mask = sample["mask"]

        return satellite_image, mask

    def __len__(self):
        return len(self.labeled_images)
    
    
    
    
# à voir si on met les dataset dans des .py séparés par la suite ? 
class ChangeDetectionS2LookingDataset(Dataset):
    """
    Custom Dataset class.
    """
    def __init__(
        self,
        list_paths_image1: List,
        list_paths_image2: List,
        list_paths_labels: List,
        transforms: Optional[Compose] = None
    ):
        """
        Constructor.

        Args:
            list_paths_image1 (List): list of path of the before state pictures
            list_paths_image2 (List): list of paths containing  the "after" state pictures
            list_paths_labels (List): list of paths containing the labeled differences (mostly segmentation masque showing the differencer between image 1 and image 2) 
        """
        self.list_paths_image1 = list_paths_image1
        self.list_paths_image2 = list_paths_image2
        self.list_paths_labels = list_paths_labels    
        self.transforms = transforms
        
    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        pathim1 = self.list_paths_image1[idx]
        pathim2 = self.list_paths_image2[idx]
        pathlabel = self.list_paths_labels[idx]
        
        label = 0
        compteur = 0
        while(np.max(label) == 0 and compteur < 15):
            cdtriplet = ChangedetectionTripletS2Looking(pathim1,pathim2,pathlabel)
            cdtriplet.random_crop(256)
            label = np.array(cdtriplet.label)
            label[label!=0] = 1
            compteur += 1
        
        img1 = np.array(cdtriplet.image1)
        img2 = np.array(cdtriplet.image2)
        
        if self.transforms:
            sample = self.transforms(image = img1, image2 = img2, mask = label)
            img1 = sample['image']
            img2 = sample['image2']
            label = sample['mask']
        else:
            img1 = torch.tensor(np.transpose(img1,(2,0,1)))
            img2 = torch.tensor(np.transpose(img2,(2,0,1)))
        
        img_double =torch.concatenate([img1,img2],axis = 0).squeeze()
        
        img_double = img_double.type(torch.float)
        
        label = torch.tensor(label)
        label = label.type(torch.LongTensor)
        
        return img_double, label, {"pathim1" : pathim1, "pathim2" : pathim2, "pathlabel" : pathlabel}
      
    def __len__(self):
        return len(self.list_paths_image1)
    



class PleiadeDataset(Dataset):
    """
    Custom Dataset class.
    """
    def __init__(
        self,
        list_paths_images: List,
        list_paths_labels: List,
        transforms: Optional[Compose] = None
    ):
        """
        Constructor.

        Args:
            list_paths_images (List): list of path of the images
            list_paths_labels (List): list of paths containing the labels 
            transforms (Compose) : list of transforms
        """
        self.list_paths_images = list_paths_images
        self.list_paths_labels = list_paths_labels    
        self.transforms = transforms
    
    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        pathim = self.list_paths_images[idx]
        pathlabel = self.list_paths_labels[idx]
        
        img = SatelliteImage.from_raster(
            file_path = pathim,
            dep = None,
            date = None,
            n_bands = 3
        ).array
        
        label = np.load(pathlabel)
        
        if self.transforms:
            sample = self.transforms(image = img, label = label)
            img = sample['image']
            label = sample['label']
        else:
            img = torch.tensor(img.astype(float))
            label = torch.tensor(label)
        
        
        img = img.type(torch.float)
        label = label.type(torch.LongTensor)
        #print(label)
        return img, label, {"pathimage" : pathim, "pathlabel" : pathlabel}
      
    def __len__(self):
        return len(self.list_paths_images)

