import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from time import time
import sys
import shutil
import numpy as np

import skimage
import skimage.io
import skimage.transform


from dataloaders import dataset
from dataloaders.dataset import getPredictionTestFileLists
from dataloaders.dataset import SCDLDataSetEnum
from utils.predict_args import getPredictionArgs
from utils.multadds_count import count_parameters_in_MB, comp_multadds
from models.stereo_model import StereoModel


class SCDLPredict():
    def __init__(self, file_path):
        self.file_path = file_path        
        self.test_ds = None        

        self.options = getPredictionArgs()

        self.kwargs = {'num_workers': self.options.threads, 'pin_memory': True, 'drop_last':True}

        print("Prediction options")
        print("#############################")
        print(self.options)

        cuda = self.options.cuda

        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")        

        self.model = StereoModel(self.options)

        torch.manual_seed(self.options.seed)
        if cuda:
            torch.cuda.manual_seed(self.options.seed)

        print('Total Params = %.2fMB' % count_parameters_in_MB(self.model))
        print('Feature Net Params = %.2fMB' % count_parameters_in_MB(self.model.feature))
        print('Matching Net Params = %.2fMB' % count_parameters_in_MB(self.model.matching))

        mult_adds = comp_multadds(self.model, input_size=(3,self.options.crop_height, self.options.crop_width)) #(3,192, 192))
        print("compute_average_flops_cost = %.2fMB" % mult_adds)

        if cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()        


        if self.options.resume:
            if os.path.isfile(self.options.resume):
                print("=> loading checkpoint '{}'".format(self.options.resume))
                checkpoint = torch.load(self.options.resume)
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                print("=> no checkpoint found at '{}'".format(self.options.resume))
        

    def predict(self, leftImage, rightImage, height, width, saveFileAndPath):
        print(saveFileAndPath)
        input1 = Variable(leftImage, requires_grad = False)
        input2 = Variable(rightImage, requires_grad = False)

        self.model.eval()

        if self.options.cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()

        start_time = time()
        with torch.no_grad():
            prediction = self.model(input1, input2)
        
        end_time = time()

        print("Processing time: {:.4f}".format(end_time - start_time))

        temp = prediction.cpu()
        temp = temp.detach().numpy()
        if height <= self.options.crop_height or width <= self.options.crop_width:
            temp = temp[0, self.options.crop_height - height: self.options.crop_height, self.options.crop_width - width: self.options.crop_width]
        else:
            temp = temp[0, :, :]

        skimage.io.imsave(saveFileAndPath, (temp * 256).astype('uint16'))


    def loadData(self):
        print("Loading data set")
        self.test_list = getPredictionTestFileLists(self.file_path)                
        self.test_ds = dataset.StereoDataset(self.options, self.file_path, self.test_list, crop_size=[self.options.crop_height,self.options.crop_width], ds_type=SCDLDataSetEnum.PREDICT)        
        print("Data set loaded")


    def runPredictor(self):

        count = len(self.test_ds)
        index = 0

        while index < count:
            input1, input2, height, width, file_name = self.test_ds[index]        
            self.predict(input1, input2, height, width, self.options.save_path + file_name)
            index+= 1        


if __name__ == '__main__':        
    predictor = SCDLPredict('/home/lbarnett/development/scdl/data/KITTI2015/testing/')
    
    predictor.loadData() 
    predictor.runPredictor()
    

    

    
