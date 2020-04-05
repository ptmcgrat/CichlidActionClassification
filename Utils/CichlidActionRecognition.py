import os
import sys
import json
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import pdb
import pandas as pd
import time

from Utils import DANN_model
from Utils.DataPrepare import DP_worker
from Utils.utils import Logger,AverageMeter, calculate_accuracy
from Utils.transforms import (Compose, Normalize, Scale, CenterCrop, 
                              RandomHorizontalFlip,RandomVerticalFlip, 
                              FixedScaleRandomCenterCrop, MultiScaleRandomCenterCrop,
                              ToTensor,TemporalCenterCrop, TemporalCenterRandomCrop,
                              ClassLabel, VideoID,TargetCompose)

from Utils.data_loader import cichlids


class ML_model():
    def __init__(self, args):
        self.args = args
        #prepare the data is the data is not prepared
        self.json_file = os.path.join(args.Log_directory,'cichlids.json')
        #check if data preparation is done
        if not os.path.exists(self.json_file):
            dp_worker = DP_worker(args)
            dp_worker.work()
        
        
    def work(self):
        
        opt = self.args
        with open(opt.Log, 'w') as opt_file:
            json.dump(vars(opt), opt_file)
        model = DANN_model.DANN_resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        parameters = model.parameters()
        criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()

        criterion = criterion.cuda()
        domain_criterion.cuda()


        source_annotateData = pd.read_csv(opt.ML_labels, sep = ',', header = 0)
        source_annotation_dict = dict(zip(source_annotateData['Location'],source_annotateData['MeanID']))
        
        target_annotateFile = os.path.join(opt.Log_directory,'target_domain_annotation.csv')
        target_annotateData = pd.read_csv(target_annotateFile, sep = ',', header = 0)
        target_annotation_dict = dict(zip(target_annotateData['Location'],target_annotateData['MeanID']))
        
        # training data loader
        
        crop_method = MultiScaleRandomCenterCrop([0.99,0.97,0.95,0.93,0.91],opt.sample_size)
        spatial_transforms = {}
        mean_file = os.path.join(opt.Log_directory,'source_Means.csv')
        with open(mean_file) as f:
            for i,line in enumerate(f):
                if i==0:
                    continue
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]]) 
                spatial_transforms[tokens[0]] = Compose([crop_method, RandomVerticalFlip(),RandomHorizontalFlip(), ToTensor(1), norm_method])
        
        temporal_transform = TemporalCenterRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        
        training_data = cichlids(opt.Clips_temp_directory,
                                 self.json_file,
                                 'training',
                                 spatial_transforms=spatial_transforms,
                                 temporal_transform=temporal_transform,
                                 target_transform=target_transform, 
                                 annotationDict =source_annotation_dict)
        
        train_loader = torch.utils.data.DataLoader(training_data,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.n_threads,
                                                   pin_memory=True)
        train_logger = Logger(os.path.join(opt.Performance_directory, 'train.log'),
            ['epoch', 'loss','domain_loss', 'train_label_acc','train_domain_acc','target_domain_acc', 'lr','alpha'])
        
        
        # validation data loader
        crop_method = CenterCrop(opt.sample_size)
        spatial_transforms = {}
        with open(mean_file) as f:
            for i,line in enumerate(f):
                if i==0:
                    continue
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]]) 
                spatial_transforms[tokens[0]] = Compose([crop_method, ToTensor(1), norm_method])
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        validation_data = cichlids(opt.Clips_temp_directory,
                                   self.json_file,
                                   'validation',
                                   spatial_transforms=spatial_transforms,
                                   temporal_transform=temporal_transform,
                                   target_transform=target_transform, 
                                   annotationDict =source_annotation_dict)
                                     
        val_loader = torch.utils.data.DataLoader(validation_data,
                                                        batch_size=opt.batch_size,
                                                        shuffle=True,
                                                        num_workers=opt.n_threads,
                                                        pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.Performance_directory, 'val.log'), ['epoch', 'loss', 'acc'])

        # test data loader
        test_data = cichlids(opt.Clips_temp_directory,
                             self.json_file,
                             'testing',
                             spatial_transforms=spatial_transforms,
                             temporal_transform=temporal_transform,
                             target_transform=target_transform, 
                             annotationDict =source_annotation_dict)
        pdb.set_trace()
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True,
                                                  num_workers=opt.n_threads,
                                                  pin_memory=True)
        test_logger = Logger(
            os.path.join(opt.Performance_directory, 'test.log'), ['epoch', 'loss', 'acc'])
        
        # target data loader
        
        spatial_transforms = {}
        mean_file = os.path.join(opt.Log_directory,'target_Means.csv')
        with open(mean_file) as f:
            for i,line in enumerate(f):
                if i==0:
                    continue
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]]) 
                spatial_transforms[tokens[0]] = Compose([crop_method, ToTensor(1), norm_method])
        target_data = cichlids(opt.Clips_temp_directory,
                               self.json_file,
                               'target',
                               spatial_transforms=spatial_transforms,
                               temporal_transform=temporal_transform,
                               target_transform=target_transform, 
                               annotationDict =target_annotation_dict)
        target_loader = torch.utils.data.DataLoader(target_data,
                                                    batch_size=opt.batch_size,
                                                    shuffle=True,
                                                    num_workers=opt.n_threads,
                                                    pin_memory=True)
        
        
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
        
        previous_domain_accuracy=0.5
        pdb.set_trace()
        for i in range(opt.n_epochs + 1):
            
            domain_average_acc,training_loss = self._train_epoch(i, train_loader, target_loader, model, criterion,domain_criterion, optimizer, opt,
                        train_logger,previous_domain_accuracy)
            previous_domain_accuracy = domain_average_acc
            
            validation_loss = self._val_epoch(i, val_loader, model, criterion, opt,val_logger)
            
            scheduler.step(training_loss)
            test_loss = self._test_epoch(i, test_loader, model, criterion, opt,test_logger)

        


    
        
    def _train_epoch(self, epoch, train_loader,target_loader, model, criterion, domain_criterion,optimizer, opt,
                epoch_logger,previous_domain_accuracy):
        print('train at epoch {}'.format(epoch))
    
        len_train = len(train_loader)
        len_target = len(target_loader)
    
        target_iter = iter(target_loader)
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        domain_losses = AverageMeter()
        train_label_accuracies = AverageMeter()
        train_domain_accuracies = AverageMeter()
        target_domain_accuracies = AverageMeter()
    
    
        end_time = time.time()
    
    
        for i, (inputs, targets) in enumerate(train_loader):
    
            p = float(i + epoch * len_train) / opt.n_epochs / len_train
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
    #         alpha = 2*previous_domain_accuracy-0.9
    #         if alpha < 0:
    #             alpha = 0
        
            data_time.update(time.time() - end_time)
            batch_size = inputs.size(0)
            targets = targets.cuda(async=True)

            inputs = Variable(inputs)
            targets = Variable(targets)
            
            train_output_label,train_output_domain = model(inputs, alpha=alpha)
            train_label_loss = criterion(train_output_label, targets)
            train_label_acc = calculate_accuracy(train_output_label, targets)
            
            train_domain_targets = torch.zeros(batch_size).long().cuda()
            train_domain_loss = domain_criterion(train_output_domain,train_domain_targets)
            train_domain_acc = calculate_accuracy(train_output_domain,train_domain_targets)
            
            train_label_accuracies.update(train_label_acc, batch_size)
            train_domain_accuracies.update(train_domain_acc, batch_size)
            
            if i < len_target:
                target_inputs,target_targets = target_iter.next()
                target_inputs = Variable(target_inputs)
                target_output_label,target_output_domain = model(target_inputs, alpha=alpha)

                target_domain_label = torch.ones(batch_size).long().cuda()
                target_domain_loss = domain_criterion(target_output_domain,target_domain_label)
                target_domain_acc = calculate_accuracy(target_output_domain,target_domain_label)
                
                target_domain_accuracies.update(target_domain_acc, batch_size)
                
                domain_loss = train_domain_loss+target_domain_loss
                loss = train_label_loss+domain_loss
                
               
            
            else:
                domain_loss = train_domain_loss
                loss = train_label_loss+train_domain_loss

            losses.update(loss.item(), batch_size)
            domain_losses.update(domain_loss.item(), batch_size)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

#             batch_logger.log({
#                 'epoch': epoch,
#                 'batch': i + 1,
#                 'iter': (epoch - 1) * len(train_loader) + (i + 1),
#                 'loss': losses.val,
#                 'train_label_acc': train_label_accuracies.val,
#                 'train_domain_acc': train_domain_accuracies.val,
#                 'target_domain_acc': target_domain_accuracies.val,
#                 'lr': optimizer.param_groups[0]['lr']})

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'train_label_Acc {train_acc.val:.3f} ({train_acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      train_acc=train_label_accuracies))
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'domain_loss':domain_losses.avg,
            'train_label_acc': train_label_accuracies.avg,
            'train_domain_acc': train_domain_accuracies.avg,
            'target_domain_acc': target_domain_accuracies.avg,
            'lr': optimizer.param_groups[0]['lr'],
            'alpha':alpha
        })

        if epoch % opt.checkpoint == 0:
            save_file_path = os.path.join(opt.Model_directory,
                                          'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        domain_average_acc = (train_domain_accuracies.avg+target_domain_accuracies.avg)/2
    
        return domain_average_acc,losses.avg
    
    
    def _val_epoch(self,epoch, data_loader, model, criterion, opt, logger):
        print('validation at epoch {}'.format(epoch))
    
        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
    
        #########  temp line, needs to be removed##################################
        file  = 'epoch_'+ str(epoch)+'_validation_matrix.csv'
        confusion_matrix = np.zeros((opt.n_classes,opt.n_classes))
        confidence_for_each_validation = {}
        ###########################################################################

        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.cuda(async=True)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs,_ = model(inputs,alpha=1)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)
                #########  temp line, needs to be removed##################################
                for j in range(len(targets)):
                    confidence_for_each_validation[paths[j]] = [x.item() for x in outputs[j]]
            
                rows = [int(x) for x in targets]
                columns = [int(x) for x in np.argmax(outputs.cpu(),1)]
                assert len(rows) == len(columns)
                for idx in range(len(rows)):
                    confusion_matrix[rows[idx]][columns[idx]] +=1
            
                ###########################################################################
                losses.update(loss.item(), inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch,
                          i + 1,
                          len(data_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))
        #########  temp line, needs to be removed##################################
        print(confusion_matrix)
        confusion_matrix = pd.DataFrame(confusion_matrix)
        confusion_matrix.to_csv(opt.Log_directory + '/ConfusionMatrix_' + str(epoch) + '.csv')
        confidence_matrix = pd.DataFrame.from_dict(confidence_for_each_validation, orient='index')
        confidence_matrix.to_csv(opt.Log_directory + '/ConfidenceMatrix.csv')
    
        #########  temp line, needs to be removed##################################
    
    
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

        return losses.avg
        
    def _test_epoch(self,epoch, data_loader, model, criterion, opt, logger):
        print('test at epoch {}'.format(epoch))
    
        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            targets = targets.cuda(async=True)
            with torch.no_grad():
            
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs,_ = model(inputs,alpha=1)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch,
                          i + 1,
                          len(data_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))

    
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

        return losses.avg
    