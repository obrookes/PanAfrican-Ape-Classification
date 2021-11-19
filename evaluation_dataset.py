import os
import glob
import json
import pickle
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PanAfricanDatasetEvaluation(Dataset):
    def __init__(self, split_path, data_path, ann_path, transforms, classes):
        
        self.transforms = transforms
        
        classes = open(classes).readlines()
        self.classes = [x.strip('\n') for x in classes]
        
        def list_split_videos(split_path):
            self.videos = open(split_path).read().split()
            """Input split file (i.e. train.txt), return list of videos ['video1', 'video1'] """
            return self.videos

        def list_path_to_anns(ann_path, videos):
            """Input path to annotations directory and list of videos, return list of abs paths to videos"""
            self.video_paths = [f"{ann_path}/{video}" for video in videos]
            return self.video_paths

        def list_annotations(video_path):
            """Input full path to video directory ('path/to/video'), return list of XML annotation files ('ann.xml') """
            self.annotations = glob.glob(f"{video_path}/*.xml")
            return self.annotations

        def get_dir_name(directory_path):
            """Input directory path ("/path/to/dir") return 'dir'"""
            return directory_path.split('/')[-1]

        def get_img_filename(annotation_path):
            """Input full path to XML file ('/path/to/file.xml'), return only corresponding JPG file ('file.jpg')"""
            return f"{annotation_path.split('/')[-1].rstrip('.xml')+'.jpg'}"

        def get_coords(obj):
            xmin = obj.find('bndbox').find('xmin').text
            ymin = obj.find('bndbox').find('ymin').text
            xmax = obj.find('bndbox').find('xmax').text
            ymax = obj.find('bndbox').find('ymax').text
            return float(xmin), float(ymin), float(xmax), float(ymax)

        def get_confidence(obj):
            return float(obj.find('confidence').text)

        def get_species(obj):    
            return obj.find('name').text
    
        videos = list_split_videos(split_path)
        video_paths = list_path_to_anns(ann_path, videos)

        self.detections = []
        self.valid_videos = []
        self.invalid_videos = []
        self.invalid_files = []

        for directory in video_paths:

            dir_name = get_dir_name(directory)
            ann_list = list_annotations(directory)

            for ann in ann_list:
                img_filename = get_img_filename(ann)

                # Construct full image path
                full_filename = f"{data_path}/{dir_name}/{dir_name}_{'_'.join(img_filename.split('_')[1:])}"

                if(os.path.isfile(full_filename)):        
                    
                    self.valid_videos.append(dir_name)
                    
                    # Open XML file
                    frame_xml = open(ann)
                    tree = ET.parse(frame_xml)
                    root = tree.getroot()

                    # Get bboxs for each detection
                    for ann in root.iter('object'):

                        det = {}

                        det['file'] = full_filename
                        det['bbox'] = list(get_coords(ann))
                        det['label'] = get_species(ann)
                        det['video'] = dir_name

                        self.detections.append(det)
                else:
                    self.invalid_videos.append(dir_name)
                    self.invalid_files.append(full_filename)

    def list_invalid_files(self):
        return self.invalid_files                      

    def list_invalid_videos(self):
        return self.invalid_videos
                    
    def list_valid_videos(self):
        return self.valid_videos
    
    def list_detections(self):
        return self.detections
    
    def sample_count(self):
        return len(self.detections)
    
    def get_detection(self, index):
        return self.detections[index]

    def __getitem__(self, index):
        
        # Get image
        img_path = self.detections[index]['file'] 
        img = Image.open(img_path)
        
        # Get bbox
        bbox = tuple(self.detections[index]['bbox'])
        
        # Labels (In my case, I only one class: target class or background)
        label = self.detections[index]['label']

        if(label=='gorilla'):
            label = 'western gorilla'
        
        tensor_label = torch.tensor(self.classes.index(label))
        
        cropped_img = img.crop(tuple(bbox))

        if self.transforms is not None:
            cropped_img = self.transforms(cropped_img)

        return cropped_img, bbox, tensor_label, img_path

    def __len__(self):
        return len(self.detections)
