import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import numpy as np
import random
import os

def getTrainingTestFileLists(file_path, ds_split):
        file_list = np.array([])
        leftImageFileName = file_path + 'image_2/'
        dirList = os.listdir(leftImageFileName)

        for file in dirList:
            if "10.png" in file:
                file_list = np.append(file_list, file)                

        np.random.shuffle(file_list)

        ds_len = len(file_list)
        train_len = int(ds_len * ds_split)

        training, test = file_list[:train_len], file_list[train_len:]

        return training, test

class StereoDataset(Dataset):
    def __init__(self, args, file_path, file_list, crop_size=[256, 256], training=True, left_right=False, shift=0):
        super(StereoDataset, self).__init__()
        self.file_path = file_path
        self.file_list = file_list
        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.left_right = left_right
        self.shift = shift                     

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index >= len(self.file_list):
            return None
        else:
            temp_data =self.loadDataPoint(self.file_list[index])

            if self.training:
                input1, input2, target = self.trainTransform(temp_data)
                return input1, input2, target
            else:
                input1, input2, target = self.testTransform(temp_data)
                return input1, input2, target
        

    def loadDataPoint(self, file_name):
        leftImageFile = self.file_path + 'image_2/' + file_name
        rightImageFile = self.file_path + 'image_3/' + file_name
        dispImageFile = self.file_path + 'disp_noc_0/' + file_name

        left = Image.open(leftImageFile)
        right = Image.open(rightImageFile)
        dispLeft = Image.open(dispImageFile)

        temp = np.asarray(dispLeft)
        size = np.shape(left)

        height = size[0]
        width = size[1]

        tempData = np.zeros([8, height, width], 'float32')
        left = np.asarray(left)
        right = np.asarray(right)
        dispLeft = np.asarray(dispLeft)

        r = left[:, :, 0]
        g = left[:, :, 1]
        b = left[:, :, 2]
 
        tempData[0, :, :] = (r-np.mean(r[:])) / np.std(r[:])
        tempData[1, :, :] = (g-np.mean(g[:])) / np.std(g[:])
        tempData[2, :, :] = (b-np.mean(b[:])) / np.std(b[:])
    
        r=right[:, :, 0]
        g=right[:, :, 1]
        b=right[:, :, 2]	

        tempData[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
        tempData[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
        tempData[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])

        tempData[6: 7, :, :] = width * 2
        tempData[6, :, :] = dispLeft[:, :]
        temp = tempData[6, :, :]
    
        temp[temp < 0.1] = width * 2 * 256
        tempData[6, :, :] = temp / 256.
    
        return tempData

    def trainTransform(self, temp_data):
        _, h, w = np.shape(temp_data)
        
        if h > self.crop_height and w <= self.crop_width:
            temp = temp_data
            temp_data = np.zeros([8, h+self.shift, self.crop_width + self.shift], 'float32')
            temp_data[6:7,:,:] = 1000
            temp_data[:, h + self.shift - h: h + self.shift, self.crop_width + self.shift - w: self.crop_width + self.shift] = temp
            _, h, w = np.shape(temp_data)
    
        if h <= self.crop_height and w <= self.crop_width:
            temp = temp_data
            temp_data = np.zeros([8, self.crop_height + self.shift, self.crop_width + self.shift], 'float32')
            temp_data[6: 7, :, :] = 1000
            temp_data[:, self.crop_height + self.shift - h: self.crop_height + self.shift, self.crop_width + self.shift - w: self.crop_width + self.shift] = temp
            _, h, w = np.shape(temp_data)
        if self.shift > 0:
            start_x = random.randint(0, w - self.crop_width)
            shift_x = random.randint(-self.shift, self.shift)
            if shift_x + start_x < 0 or shift_x + start_x + self.crop_width > w:
                shift_x = 0
            start_y = random.randint(0, h - self.crop_height)
            left = temp_data[0: 3, start_y: start_y + self.crop_height, start_x + shift_x: start_x + shift_x + self.crop_width]
            right = temp_data[3: 6, start_y: start_y + self.crop_height, start_x: start_x + self.crop_width]
            target = temp_data[6: 7, start_y: start_y + self.crop_height, start_x + shift_x : start_x+shift_x + self.crop_width]
            target = target - shift_x
            return left, right, target
        if h <= self.crop_height and w <= self.crop_width:
            temp = temp_data
            temp_data = np.zeros([8, self.crop_height, self.crop_width], 'float32')
            temp_data[:, self.crop_height - h: self.crop_height, self.crop_width - w: self.crop_width] = temp
        else:
            start_x = random.randint(0, w - self.crop_width)
            start_y = random.randint(0, h - self.crop_height)
            temp_data = temp_data[:, start_y: start_y + self.crop_height, start_x: start_x + self.crop_width]
        if random.randint(0, 1) == 0 and self.left_right:
            right = temp_data[0: 3, :, :]
            left = temp_data[3: 6, :, :]
            target = temp_data[7: 8, :, :]
            return left, right, target
        else:
            left = temp_data[0: 3, :, :]
            right = temp_data[3: 6, :, :]
            target = temp_data[6: 7, :, :]
            return left, right, target    

    def testTransform(self, temp_data):
        _, h, w = np.shape(temp_data)

        if h <= self.crop_height and w <= self.crop_width:
            temp = temp_data
            temp_data = np.zeros([8,self.crop_height,self.crop_width], 'float32')
            temp_data[6: 7, :, :] = 1000
            temp_data[:, self.crop_height - h: self.crop_height, self.crop_width - w: self.crop_width] = temp
        else:
            start_x = (w-self.crop_width)//2
            start_y = (h-self.crop_height)//2
            temp_data = temp_data[:, start_y: start_y + self.crop_height, start_x: start_x + self.crop_width]
    
        left = temp_data[0: 3, :, :]
        right = temp_data[3: 6, :, :]
        target = temp_data[6: 7, :, :]

        return left, right, target

    def makeDataLoaders(self, args):

        ds_size = len(self)
        train_size = int(ds_size * 0.8)
        valid_size = ds_size - train_size

        train_loader, valid_loader = random_split(self, [train_size, valid_size])        
        return train_loader, valid_loader