import os,sys,json,torch,torchvision,pdb,time, scipy
from torch import nn,optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import pandas as pd
import numpy as np
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
        if opt.Purpose =='classify':
            source_annotateData = pd.read_csv(opt.Videos_to_project_file, sep = ',', header = 0)
        else:
            source_annotateData = pd.read_csv(opt.ML_labels, sep = ',', header = 0)
        #pdb.set_trace()
        source_annotation_dict = dict(zip(source_annotateData['ClipName'].str.replace('.mp4',''),source_annotateData['ProjectID']))
        
        
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
        if opt.Purpose == 'classify':
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
            validation_data = cichlids(opt.Temporary_clips_directory,
                                    self.source_json_file,
                                    'validation',
                                    spatial_transforms=spatial_transforms,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform, 
                                    annotationDict =source_annotation_dict,
                                    args = self.args)
                                        
            val_loader = torch.utils.data.DataLoader(validation_data,
                                                            batch_size=opt.batch_size,
                                                            shuffle=False,
                                                            num_workers=opt.n_threads,
                                                            pin_memory=True)
            val_logger = Logger(
                os.path.join(opt.Results_directory, 'val.log'), ['epoch', 'loss', 'acc'])
            
        else:
            # pdb.set_trace()
            training_data = cichlids(opt.Temporary_clips_directory,
                                    self.source_json_file,
                                    'training',
                                    spatial_transforms=spatial_transforms,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform, 
                                    annotationDict =source_annotation_dict,
                                    args = self.args)
            training_data[0]
            if len(training_data) != 0:
                train_loader = torch.utils.data.DataLoader(training_data,
                                                        batch_size=opt.batch_size,
                                                        shuffle=True,
                                                        num_workers=opt.n_threads,
                                                        pin_memory=True)
                train_logger = Logger(
                    os.path.join(opt.Results_directory, 'train.log'),
                    ['epoch', 'loss', 'acc', 'lr'])
                #train_batch_logger = Logger(
                #    os.path.join(opt.Results_directory, 'train_batch.log'),
                #    ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
            
            else:
                train_loader = None
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
            validation_data = cichlids(opt.Temporary_clips_directory,
                                    self.source_json_file,
                                    'validation',
                                    spatial_transforms=spatial_transforms,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform, 
                                    annotationDict =source_annotation_dict,
                                    args = self.args)
                                        
            val_loader = torch.utils.data.DataLoader(validation_data,
                                                            batch_size=opt.batch_size,
                                                            shuffle=False,
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
            test_data = cichlids(opt.Temporary_clips_directory,
                                    self.source_json_file,
                                    'testing',
                                    spatial_transforms=spatial_transforms,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    annotationDict=source_annotation_dict,args = self.args)
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

        if opt.Purpose in ['finetune','classify']:
            checkpoint = torch.load(opt.Trained_model)
            begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            begin_epoch = 0
        if opt.Purpose == 'classify':
            _,confusion_matrix,confidence_matrix, results_df = self.val_epoch(i, val_loader, model, criterion, opt, val_logger)
            with open(self.source_json_file,'r') as input_f:
                source_json = json.load(input_f)
            confidence_matrix.columns = source_json['labels']
            confidence_matrix['predicted_label'] = confidence_matrix.idxmax(axis="columns")
            confidence_matrix.to_csv(self.args.Output_file)
            # pdb.set_trace()
            return
        print('run')
        # pdb.set_trace()
        if train_loader is not None: 
            for i in range(begin_epoch,opt.n_epochs + 1):
                self.train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger)

                validation_loss,confusion_matrix,p_dt,results_df = self.val_epoch(i, val_loader, model, criterion, opt, val_logger)
                
                confusion_matrix_file = os.path.join(self.args.Results_directory,'epoch_{epoch}_confusion_matrix.csv'.format(epoch=i))
                confusion_matrix.to_csv(confusion_matrix_file)
                validation_results_file = os.path.join(self.args.Results_directory,'epoch_{epoch}_results.csv'.format(epoch=i))
                s_dt = pd.read_csv(self.args.ML_labels, index_col = 0)
                s_dt['Location'] = s_dt.ClipName.str.replace('.mp4','')
                g_dt = pd.merge(p_dt,s_dt, left_index=True, right_on = 'Location')
                g_dt = g_dt[['Location','AnalysisID','Probability']]
                results_df.to_csv(validation_results_file)
                results_df['Match'] = results_df.TrueLabel == results_df.PredictedLabel
                out_dt = pd.merge(g_dt,results_df, left_on = 'Location', right_on = 'ClipName')
                pdb.set_trace()
                out_dt.groupby('AnalysisID').agg({'Match':'mean','Match':'count'})
                scheduler.step(validation_loss)
                if i % 5 == 0 and len(test_data) != 0:
                    _ = self.val_epoch(i, test_loader, model, criterion, opt, test_logger)


    def train_epoch(self, epoch, data_loader, model, criterion, optimizer, opt,
                    epoch_logger, batch_logger):
        print('train at epoch {}'.format(epoch))
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        # pdb.set_trace()
        for i, (inputs, targets,_) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.cuda(non_blocking=True)
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
            save_file_path = os.path.join(opt.Results_directory,
                                          'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
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
        confusion_matrix = np.zeros((opt.n_classes,opt.n_classes))
        confidence_for_each_validation = {}
        ###########################################################################
        results =[]
        # pdb.set_trace()
        for i, (inputs, targets,paths) in enumerate(data_loader):
            # pdb.set_trace()
            data_time.update(time.time() - end_time)

            targets = targets.cuda(non_blocking=True)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = calculate_accuracy(outputs, targets)
                ########  temp line, needs to be removed##################################

                predictedLabel = torch.argmax(outputs,dim =1).cpu().numpy()
                targetLabel = targets.cpu().numpy()

                for j in range(len(targets)):
                    key = paths[j].split('/')[-1]
                    confidence_for_each_validation[key] = [x.item() for x in outputs[j]]
                    results.append({"ClipName":key, "TrueLabel":targetLabel[j],"PredictedLabel":predictedLabel[j]})

                    
                rows = [int(x) for x in targets]
                columns = [int(x) for x in np.argmax(outputs.data.cpu(),1)]
                assert len(rows) == len(columns)
                for idx in range(len(rows)):
                    confusion_matrix[rows[idx]][columns[idx]] +=1

                ###########################################################################
                losses.update(loss.data, inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()
                

                # pdb.set_trace()

                

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
        confusion_matrix = pd.DataFrame(confusion_matrix)
            # confusion_matrix.to_csv(file)
        probability_matrix = {k: scipy.special.softmax(v).max() for k,v in confidence_for_each_validation.items()}
        confidence_matrix = pd.DataFrame.from_dict(probability_matrix, orient='index', columns = ['Probability'])
        results_df = pd.DataFrame(results)
        # confidence_matrix.to_csv('confidence_matrix.csv')

            #########  temp line, needs to be removed##################################

        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

        return losses.avg,confusion_matrix, confidence_matrix, results_df
        
    def test_epoch(self, epoch, data_loader, model, criterion, opt, logger):
        print('test at epoch {}'.format(epoch))

        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        for i, (inputs, targets,_) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            if not opt.no_cuda:
                targets = targets.cuda(non_blocking=True)
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
    
