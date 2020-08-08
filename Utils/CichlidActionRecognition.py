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

from Utils.model import resnet18
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
        self.source_json_file = os.path.join(args.Results_directory,'source.json')

    def work(self):
        opt = self.args
        log_file = os.path.join(opt.Results_directory,'log')
        with open(log_file, 'w') as output:
            json.dump(vars(opt), output)
        model = resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        parameters = model.parameters()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        source_annotateData = pd.read_csv(opt.ML_labels, sep = ',', header = 0)
        source_annotation_dict = dict(zip(source_annotateData['Location'],source_annotateData['MeanID']))
        
        # training data loader
        
        crop_method = MultiScaleRandomCenterCrop([0.99,0.97,0.95,0.93,0.91],opt.sample_size)
        spatial_transforms = {}
        mean_file = os.path.join(opt.Results_directory,'Means.csv')
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
                                 self.source_json_file,
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
        train_logger = Logger(
            os.path.join(opt.Results_directory, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.Results_directory, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        
        
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
                                   self.source_json_file,
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
            os.path.join(opt.Results_directory, 'val.log'), ['epoch', 'loss', 'acc'])

        # test data loader
        crop_method = CenterCrop(opt.sample_size)
        spatial_transforms = {}
        with open(mean_file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]])
                spatial_transforms[tokens[0]] = Compose([crop_method, ToTensor(1), norm_method])
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        test_data = cichlids(opt.Clips_temp_directory,
                                   self.source_json_file,
                                   'test',
                                   spatial_transforms=spatial_transforms,
                                   temporal_transform=temporal_transform,
                                   target_transform=target_transform,
                                   annotationDict=source_annotation_dict)
        if len(test_data) != 0:
            test_loader = torch.utils.data.DataLoader(test_data,
                                                     batch_size=opt.batch_size,
                                                     shuffle=True,
                                                     num_workers=opt.n_threads,
                                                     pin_memory=True)
            test_logger = Logger(
                os.path.join(opt.Results_directory, 'test.log'), ['epoch', 'loss', 'acc'])


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

        if opt.Purpose == 'finetune':
            checkpoint = torch.load(os.path.join(opt.Model_directory,'save_60.pth'))
            begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
#             optimizer = optim.Adam(parameters,lr=0.0001)
        else:
            begin_epoch = 0

        print('run')

        for i in range(begin_epoch,opt.n_epochs + 1):
            self.train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)

            validation_loss = self.val_epoch(i, val_loader, model, criterion, opt, val_logger)

            scheduler.step(validation_loss)


    def train_epoch(self, epoch, data_loader, model, criterion, optimizer, opt,
                    epoch_logger, batch_logger):
        print('train at epoch {}'.format(epoch))
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.cuda(async=True)
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            })

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
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

            if epoch % opt.checkpoint == 0:
                save_file_path = os.path.join(opt.result_path,
                                              'save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)

    def val_epoch(self, epoch, data_loader, model, criterion, opt, logger):
        print('validation at epoch {}'.format(epoch))

        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        #########  temp line, needs to be removed##################################
        #     file  = 'epoch_'+ str(epoch)+'_validation_matrix.csv'
        #     confusion_matrix = np.zeros((opt.n_classes,opt.n_classes))
        #     confidence_for_each_validation = {}
        ###########################################################################

        # pdb.set_trace()
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.cuda(async=True)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)
                #########  temp line, needs to be removed##################################
                #             for j in range(len(targets)):
                #                 confidence_for_each_validation[paths[j]] = [x.item() for x in outputs[j]]
                '''
                rows = [int(x) for x in targets]
                columns = [int(x) for x in np.argmax(outputs,1)]
                assert len(rows) == len(columns)
                for idx in range(len(rows)):
                    confusion_matrix[rows[idx]][columns[idx]] +=1
                '''
                ###########################################################################
                losses.update(loss.data, inputs.size(0))
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
            # print(confusion_matrix)
            # confusion_matrix = pd.DataFrame(confusion_matrix)
            # confusion_matrix.to_csv(file)
            #     confidence_matrix = pd.DataFrame.from_dict(confidence_for_each_validation, orient='index')
            #     confidence_matrix.to_csv('confidence_matrix.csv')

            #########  temp line, needs to be removed##################################

            logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

            return losses.avg
        
    def test_epoch(self, epoch, data_loader, model, criterion, opt, logger):
        print('test at epoch {}'.format(epoch))

        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            if not opt.no_cuda:
                targets = targets.cuda(async=True)
                with torch.no_grad():
                    inputs = Variable(inputs)
                    targets = Variable(targets)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    acc = calculate_accuracy(outputs, targets)
                    losses.update(loss.data, inputs.size(0))
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
    