import os
import torch
import torchmetrics
import pretrainedmodels
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import loggers
from collections import OrderedDict
from argparse import ArgumentParser
from data_module import PanAfricanDataModule

class EnsembleModel(pl.LightningModule):

    def __init__(self, num_classes, input_sizes, checkpoint, gpu):

        super().__init__()

        modelIncept = pretrainedmodels.__dict__["inceptionv4"](num_classes=1000, pretrained="imagenet")
        modelIncept.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        modelIncept.last_linear = nn.Linear(modelIncept.last_linear.in_features, num_classes)
        self.modelIncept = modelIncept


        modelResnet = pretrainedmodels.__dict__["se_resnext101_32x4d"](num_classes=1000, pretrained="imagenet")
        modelResnet.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        modelResnet.last_linear = nn.Linear(modelResnet.last_linear.in_features, num_classes)
        self.modelResnet = modelResnet


        self.input_sizes = input_sizes
        assert len(input_sizes) == 2, 'Two input resolutions need to be specified for ensembles.'
#       Loss stuff
#         if self.use_onevsall_loss:
#             assert self.label_smoothing > 0, 'One-vs-all loss requires label smoothing larger than 0.'
#             self.criterion = criterions.BCEWithLogitsLoss2().cuda()
#         elif self.label_smoothing > 0:
#             self.criterion = criterions.KLDivLoss2()
#         else:
        self.criterion = nn.CrossEntropyLoss()
    
        self.checkpoint = checkpoint
        self.gpu = gpu

    	# Initialise metrics
        self.top1_train_accuracy = torchmetrics.Accuracy(top_k=1)
        self.top3_train_accuracy = torchmetrics.Accuracy(top_k=3)

        self.top1_val_accuracy = torchmetrics.Accuracy(top_k=1)  
        self.top3_val_accuracy = torchmetrics.Accuracy(top_k=3) 

    def forward(self, x):
        input_incept = F.interpolate(x, (self.input_sizes[0], self.input_sizes[0]), mode='bilinear') 
        input_resnet = F.interpolate(x, (self.input_sizes[1], self.input_sizes[1]), mode='bilinear') 
        return (self.modelIncept(input_incept) + self.modelResnet(input_resnet)) / 2

    def configure_optimizers(self):

        optimiser = torch.optim.SGD(self.parameters(), lr=0.0045, momentum=0.9, weight_decay=0.0001)
        return optimiser

    def training_step(self, batch, batch_idx):

        data, targets = batch
        preds = self(data)
        loss = self.criterion(preds, targets)
        
        top1_train_acc = self.top1_train_accuracy(preds, targets)
        top3_train_acc = self.top3_train_accuracy(preds, targets)
         
        self.log('top1_train_acc', top1_train_acc, logger=False, on_epoch=False, on_step=True, prog_bar=True)
        self.log('top3_train_acc', top3_train_acc, logger=False, on_epoch=False, on_step=True, prog_bar=True)
        self.log('train_loss', loss, logger=True, on_epoch=True, on_step=True)
        
        return {"loss": loss, "logs": {"train_loss": loss, "top1_train_acc": top1_train_acc, "top3_train_acc": top3_train_acc}}

    def training_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_train_accuracy.compute()
        top3_acc = self.top3_train_accuracy.compute()
        self.log('train_top1_acc_epoch', top1_acc, logger=True, on_epoch=True, on_step=False, prog_bar=False)
        self.log('train_top3_acc_epoch', top3_acc, logger=True, on_epoch=True, on_step=False, prog_bar=False)  

        # Log epoch loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', loss, logger=True, on_epoch=True, on_step=False, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
  
        data, targets = batch
        preds = self(data)
        loss = self.criterion(preds, targets)

        top1_val_acc = self.top1_val_accuracy(preds, targets)
        top3_val_acc = self.top3_val_accuracy(preds, targets)
          
        self.log('top1_train_acc', top1_val_acc, logger=False, on_epoch=False, on_step=False, prog_bar=True)
        self.log('top3_train_acc', top3_val_acc, logger=False, on_epoch=False, on_step=False, prog_bar=True)
        self.log('train_loss', loss, logger=True, on_epoch=True, on_step=True)
        
        return {"loss": loss, "logs": {"val_loss": loss, "top1_val_acc": top1_val_acc, "top3_val_acc": top3_val_acc}}


    def validation_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_val_accuracy.compute()
        top3_acc = self.top3_val_accuracy.compute()
        self.log('val_top1_acc_epoch', top1_acc, logger=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_top3_acc_epoch', top3_acc, logger=True, on_epoch=True, on_step=False, prog_bar=True)  

        # Log epoch loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss_epoch', loss, logger=True, on_epoch=True, on_step=False, prog_bar=True)
 

    def load_model(self):

        """
        Loads a model from a checkpoint.
        """

        if os.path.isfile(self.checkpoint):
            print("=> loading checkpoint '{}'".format(self.checkpoint))

            if self.gpu:
                cuda_device = torch.cuda.current_device()
                checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage.cuda(cuda_device))
            else:
                checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)

            start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            best_prec1 = checkpoint['best_prec1'] if 'best_prec1' in checkpoint else 0
            best_prec3 = checkpoint['best_prec3'] if 'best_prec3' in checkpoint else 0
            best_prec5 = checkpoint['best_prec5'] if 'best_prec5' in checkpoint else 0

            state_dict = checkpoint['state_dict']
            classnames = checkpoint['classnames']
            model_type = checkpoint['model_type']

            print('Loaded %d classes' % len(classnames))

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                module = k[0:7] # check for 'module.' of dataparallel
                name = k[7:] # remove 'module.' of dataparallel            

                if k[:7] == 'module.':
                    k = k[7:]
                if k[:2] == '1.':
                    k = k[2:]
                if k[:6] == 'model.':
                    k = k[6:]

                new_state_dict[k] = v

                #print("%s" % (k))

            model_dict = new_state_dict        
            optimizer_dict = checkpoint['optimizer'] if 'optimizer' in checkpoint else None

            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(self.checkpoint, start_epoch))

            data.best_prec1 = best_prec1
            data.best_prec3 = best_prec3
            data.best_prec5 = best_prec5
            data.start_epoch = start_epoch
            data.classnames = classnames
            data.model_dict = model_dict
            data.optimizer_dict = optimizer_dict
            data.model_type = model_type

            self.load_state_dict(data.model_dict)
            print('Loaded state_dict succesfully!')

        else:
            print("=> no checkpoint found at '{}'".format(self.checkpoint))


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
    
    data_module = PanAfricanDataModule(splits=splits, data=data_path, annotation=ann_path, classes=classes, batch_size=args.batch_size, num_workers=args.num_workers)
    
    if(args.gpus > 0):
        gpu=True
    else:
        gpu=False

    model = EnsembleModel(num_classes=5266, input_sizes=[560, 560], checkpoint=checkpoint, gpu=gpu)
    model.load_model()

    # Checkpoint callbacks    
    val_acc_checkpoint = ModelCheckpoint(
        monitor="val_top1_acc_epoch",
        dirpath="/weights",
        filename="best_validation_acc_epoch={epoch}",
        mode="max"
    )

    tb_logger = loggers.TensorBoardLogger('log', name='behaviour')

    # Initialise trainer
    if(gpu):
        trainer = pl.Trainer(callbacks=[val_acc_checkpoint], gpus=args.gpus, nodes=args.nodes, strategy='ddp')
    else:
        trainer = pl.Trainer()

    # Train!
    trainer.fit(model, data_module)

if __name__=="__main__":

    parser = ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--num_workers", type=int, default=8, required=False)
    parser.add_argument("--gpus", type=int, default=8, required=False)
    parser.add_argument("--nodes", type=int, default=2, required=False)

    args = parser.parse_args()
    
    main(args)
