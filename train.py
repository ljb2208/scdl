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


from dataloaders import dataset
from dataloaders.dataset import getTrainingTestFileLists
from utils.train_args import getTrainingArgs
from utils.multadds_count import count_parameters_in_MB
from models.stereo_model import StereoModel


class SCDLTrain():
    def __init__(self, file_path):
        self.file_path = file_path
        self.train_ds = None
        self.test_ds = None

        self.train_loader = None
        self.test_loader = None

        self.options = getTrainingArgs()

        self.kwargs = {'num_workers': self.options.threads, 'pin_memory': True, 'drop_last':True}

        print("Training options")
        print("#############################")
        print(self.options)

        cuda = self.options.cuda

        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")        

        self.model = StereoModel(self.options)

        print(self.model)

        torch.manual_seed(self.options.seed)
        if cuda:
            torch.cuda.manual_seed(self.options.seed)

        print('Total Params = %.2fMB' % count_parameters_in_MB(self.model))
        print('Feature Net Params = %.2fMB' % count_parameters_in_MB(self.model.feature))
        print('Matching Net Params = %.2fMB' % count_parameters_in_MB(self.model.matching))

        if cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()

        torch.backends.cudnn.benchmark = True

        if self.options.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.options.lr, betas=(0.9,0.999))
        elif self.options.solver == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.options.lr, momentum=0.9)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.options.milestones, gamma=0.5)

        if self.options.resume:
            if os.path.isfile(self.options.resume):
                print("=> loading checkpoint '{}'".format(self.options.resume))
                checkpoint = torch.load(self.options.resume)
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                print("=> no checkpoint found at '{}'".format(self.options.resume))
        


    def train(self, epoch):                


        epoch_loss = 0
        epoch_error = 0
        valid_iteration = 0
        
        for iteration, batch in enumerate(self.train_loader):
            input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), (batch[2])
            if self.options.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda()

            target=torch.squeeze(target,1)
            mask = target < self.options.maxdisp
            mask.detach_()
            valid = target[mask].size()[0]
            train_start_time = time()
            if valid > 0:
                self.model.train()
        
                self.optimizer.zero_grad()
                disp = self.model(input1,input2) 
                loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
                loss.backward()
                self.optimizer.step()
                
                error = torch.mean(torch.abs(disp[mask] - target[mask])) 
                train_end_time = time()
                train_time = train_end_time - train_start_time

                epoch_loss += loss.item()
                valid_iteration += 1
                epoch_error += error.item()
                print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Error: ({:.4f}), Time: ({:.2f}s)".format(epoch, iteration, len(self.train_loader), loss.item(), error.item(), train_time))
                sys.stdout.flush()      

        print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Error: ({:.4f})".format(epoch, epoch_loss / valid_iteration, epoch_error/valid_iteration))

    def val(self):
        epoch_error = 0
        valid_iteration = 0
        three_px_acc_all = 0
        self.model.eval()
        for iteration, batch in enumerate(self.train_loader):
            input1, input2, target = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)
            if self.options.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
                target = target.cuda()
            target=torch.squeeze(target,1)
            mask = target < self.options.maxdisp
            mask.detach_()
            valid=target[mask].size()[0]
            if valid>0:
                with torch.no_grad(): 
                    disp = self.model(input1,input2)
                    error = torch.mean(torch.abs(disp[mask] - target[mask])) 

                    valid_iteration += 1
                    epoch_error += error.item()              
                    #computing 3-px error#                
                    pred_disp = disp.cpu().detach() 
                    true_disp = target.cpu().detach()
                    disp_true = true_disp
                    index = np.argwhere(true_disp<self.options.maxdisp)
                    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
                    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
                    three_px_acc = 1-(float(torch.sum(correct))/float(len(index[0])))

                    three_px_acc_all += three_px_acc
        
                    print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(iteration, len(self.test_loader), error.item(), three_px_acc))
                    sys.stdout.flush()

        print("===> Test: Avg. Error: ({:.4f} {:.4f})".format(epoch_error/valid_iteration, three_px_acc_all/valid_iteration))
        return three_px_acc_all/valid_iteration


    def save_checkpoint(self, save_path, epoch,state, is_best):
        filename = save_path + "epoch_{}.pth".format(epoch)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, save_path + 'best.pth')
        print("Checkpoint saved to {}".format(filename))


    def loadData(self):
        print("Loading data sets")
        train_list, test_list = getTrainingTestFileLists(self.file_path, 0.8)        
        self.train_ds = dataset.StereoDataset(self.options, self.file_path, train_list, crop_size=[self.options.crop_height, self.options.crop_width], training=True)
        self.test_ds = dataset.StereoDataset(self.options, self.file_path, test_list, crop_size=[384,1248], training=False)

        self.train_loader= DataLoader(self.train_ds, batch_size=self.options.batch_size, shuffle=False, **self.kwargs)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.options.testBatchSize, shuffle=False, **self.kwargs)
        print("Data sets loaded")


    def runTrainer(self):
        error=100
        for epoch in range(1, self.options.nEpochs + 1):
            self.train(epoch)
            is_best = False
            loss=self.val()
            if loss < error:
                error=loss
                is_best = True
            
            if epoch%100 == 0 and epoch >= 3000:
                self.save_checkpoint(self.options.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                    }, is_best)
            if is_best:
                self.save_checkpoint(self.options.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                    }, is_best)

            self.scheduler.step()

        self.save_checkpoint(self.options.save_path, self.option.nEpochs,{
                'epoch': self.opt.nEpochs,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
            }, is_best)



if __name__ == '__main__':        
    trainer = SCDLTrain('/home/lbarnett/development/scdl/data/KITTI2015/training/')

    
    trainer.loadData() 
    trainer.runTrainer()   

    

    
