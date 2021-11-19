import os
import torch
import json
import tqdm
from argparse import ArgumentParser
from torchvision import transforms
from lightning_train import EnsembleModel
from torch.utils.data import DataLoader
from evaluation_dataset import PanAfricanDatasetEvaluation

def main(args):
    
    splits = [
        '/home/dl18206/Desktop/mres/summer_project/project/data/dataset/splits/trainingdata.txt',
        '/home/dl18206/Desktop/mres/summer_project/project/data/dataset/splits/validationdata.txt',
        '/home/dl18206/Desktop/mres/summer_project/project/data/dataset/splits/testdata.txt'
        ]

    data_path = '/home/dl18206/Desktop/mres/summer_project/project/data/dataset/frames/rgb'
    ann_path = '/home/dl18206/Desktop/mres/summer_project/project/data/dataset/annotations'

    checkpoint = 'species_classification.2019.12.00.pytorch'
    
    classes = 'species_classification.2019.12.00.common_names.txt'

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = PanAfricanDatasetEvaluation(split_path=splits[1],
                                          data_path=data_path,
                                          ann_path=ann_path,
                                          transforms=transform,
                                          classes = classes)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    if(args.gpus > 0):
        gpu=True
    else:
        gpu=False
    
    model = EnsembleModel(num_classes=5266, input_sizes=[560, 560], checkpoint=checkpoint, gpu=gpu)
    model.load_model()

    class_list = open(classes).readlines()
    class_list = [c.strip('/n') for c in class_list]  

    detections = []

    for img, bbox, label, filename in tqdm.tqdm(data_loader):
        
        detection = {}
        detection['file'] = filename

        bbox = [b.item() for b in bbox]

        preds = model(img)
        pred_label = preds.topk(k=1).indices.item()
        pred_name = class_list[pred_label]
        
        det = {}
        det['bbox'] = bbox
        det['label_index'] = pred_label
        det['label'] = pred_name

        detection['detections'] = det
        detections.append(detection)

    file = {}
    file['images'] = detections

    with open(args.export_name, 'w') as fp:
        json.dump(file, fp)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("--split_index", type=int, default=1, required=False, help='Expects int 0, 1 or 2 which correspond to train, val and test')
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--num_workers", type=int, default=6, required=False)
    parser.add_argument("--gpus", type=int, default=0, required=False)
    parser.add_argument("--export_name", type=str, default='data.json', required=False, help='Name of the output JSON file')

    args = parser.parse_args()
    
    main(args)
