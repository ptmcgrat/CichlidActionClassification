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


from opts import parse_opts
from resnet import resnet18
from transforms import (
    Compose, Normalize, Scale, CenterCrop, 
    RandomHorizontalFlip, MultiScaleRandomCenterCrop, 
    ToTensor,TemporalCenterCrop, TemporalCenterRandomCrop,
    ClassLabel, VideoID,TargetCompose)
from data_loader import get_training_set, get_validation_set, get_test_set
from utils import Logger,AverageMeter, calculate_accuracy





def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, paths) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        out_str = []
        for clip_data in range(inputs.shape[0]):
            for color in range(inputs.shape[1]):
                out_str.append(inputs[clip_data][color].mean())

        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
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
            'lr': optimizer.param_groups[0]['lr'],
            'means': ','.join([str(x) for x in out_str])
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

def val_epoch(epoch, data_loader, model, criterion, opt, logger):
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

    for i, (inputs, targets, paths) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        with torch.no_grad():
            
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
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
    confusion_matrix.to_csv(opt.result_path + '/ConfusionMatrix_' + str(epoch) + '.csv')
    confidence_matrix = pd.DataFrame.from_dict(confidence_for_each_validation, orient='index')
    confidence_matrix.to_csv(opt.result_path + '/ConfidenceMatrix.csv')
    
    #########  temp line, needs to be removed##################################
    
    
    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg


def test_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('test at epoch {}'.format(epoch))
    
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    

    for i, (inputs, targets, paths) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        with torch.no_grad():
            
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
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
    return


if __name__ == '__main__':
    
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
    opt.arch = 'resnet-{}'.format(opt.model_depth)
    print(opt)
    #with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    #    json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    model = resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                t_stride=opt.t_stride)
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
    parameters = model.parameters()
#     model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    if not opt.no_train:
        crop_method = MultiScaleRandomCenterCrop(opt.sample_size)
        spatial_transforms = {}
        with open(opt.mean_file) as f:
            for i,line in enumerate(f):
                if i==0:
                    continue
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]]) 
                spatial_transforms[tokens[0]] = Compose([crop_method, RandomHorizontalFlip(), ToTensor(opt.norm_value), norm_method])
        annotateData = pd.read_csv(opt.annotation_file, sep = ',', header = 0)
        keys = annotateData[annotateData.Dataset=='Train']['Location']
        values = annotateData[annotateData.Dataset=='Train']['MeanID']

        annotationDictionary = dict(zip(keys, values))
        pdb.set_trace()
        temporal_transform = TemporalCenterRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
#         pdb.set_trace()
        training_data = get_training_set(opt, spatial_transforms,
                                         temporal_transform, target_transform, annotationDictionary)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr', 'means'])

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
    if not opt.no_val:
        spatial_transforms = {}
        with open(opt.mean_file) as f:
            for i,line in enumerate(f):
                if i==0:
                    continue
                tokens = line.rstrip().split(',')
                norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]]) 
                spatial_transforms[tokens[0]] = Compose([CenterCrop(opt.sample_size),ToTensor(opt.norm_value), norm_method])

        annotateData = pd.read_csv(opt.annotation_file, sep = ',', header = 0)
        keys = annotateData[annotateData.Dataset=='Test']['Location']
        values = annotateData[annotateData.Dataset=='Test']['MeanID']

        annotationDictionary = dict(zip(keys, values))
        

        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        #temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transforms, temporal_transform, target_transform, annotationDictionary)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if not opt.no_test:


        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_test_set(
            opt, spatial_transforms, temporal_transform, target_transform, annotationDictionary)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
        
        if not opt.no_val:
            test_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

