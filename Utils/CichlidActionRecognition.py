import os,sys,json,torch,torchvision,pdb,time, scipy
from torch import nn,optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import pandas as pd
import numpy as np
from Utils.model import resnet18
from Utils.utils import Logger,AverageMeter, calculate_accuracy

from Utils.data_loader import cichlids


class ML_model():
    def __init__(self, purpose, json_datafile, temp_clips_directory, results_directory, sample_length, sample_duration):
        self.purpose = purpose
        assert purpose in ['Train','Classify']
        self.sourceJSON = json_datafile
        self.tempClipsDir = temp_clips_directory
        self.resultsDirectory = results_directory
        #prepare the data is the data is not prepared
        self.sample_length = sample_length
        self.sample_duration = sample_duration
    
    def createDataLoaders(self, batch_size, n_threads):
        if self.purpose == 'Train':
            training_data = cichlids(self.tempClipsDir, self.sourceJSON, 'train')
            training_data.readDatabase()
            training_data.createTransforms(self.sample_length, self.sample_duration)
            self.train_loader = torch.utils.data.DataLoader(training_data,
                batch_size=batch_size,shuffle=True,num_workers=n_threads, pin_memory=True)
            self.train_logger = Logger(os.path.join(self.resultsDirectory, 'train.log'),
                    ['epoch', 'loss', 'acc', 'lr'])

        validation_data = cichlids(self.tempClipsDir, self.sourceJSON, 'validation')
        validation_data.readDatabase()
        validation_data.createTransforms(self.sample_length, self.sample_duration)
        self.val_loader = torch.utils.data.DataLoader(validation_data,
            batch_size=batch_size,shuffle=False,num_workers=n_threads, pin_memory=True)
        self.val_logger = Logger(os.path.join(self.resultsDirectory, 'val.log'),
                    ['epoch', 'loss', 'acc'])
        
    def train_model(self, n_classes, dampening, learning_rate, momentum, weight_decay, nesterov, lr_patience, n_epochs, checkpoint, ML_labels):
        self.n_classes = n_classes
        model = resnet18(
                num_classes=n_classes,
                sample_size=self.sample_length,
                sample_duration=self.sample_duration)

        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        parameters = model.parameters()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
                                  
        if nesterov:
            dampening = 0
        
        optimizer = optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov)
        
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=lr_patience)

        begin_epoch = 0
        
        for i in range(begin_epoch,n_epochs + 1):
            self.train_epoch(i, self.train_loader, model, criterion, optimizer, self.train_logger, checkpoint)

            validation_loss,confusion_matrix,p_dt,results_df = self.val_epoch(i, self.val_loader, model, criterion, self.val_logger)
            
            confusion_matrix_file = os.path.join(self.resultsDirectory,'epoch_{epoch}_confusion_matrix.csv'.format(epoch=i))
            confusion_matrix.to_csv(confusion_matrix_file)
            validation_results_file = os.path.join(self.resultsDirectory,'epoch_{epoch}_results.csv'.format(epoch=i))
            s_dt = pd.read_csv(ML_labels, index_col = 0)
            s_dt['Location'] = s_dt.ClipName.str.replace('.mp4','')
            s_dt['ProjectID'] = s_dt.ClipName.str.split('__').str[0]
            g_dt = pd.merge(p_dt,s_dt, left_index=True, right_on = 'Location')
            g_dt = g_dt[['Location','AnalysisID','ProjectID','Probability']]
            results_df.to_csv(validation_results_file)
            results_df['Match'] = results_df.TrueLabel == results_df.PredictedLabel
            out_dt = pd.merge(g_dt,results_df, left_on = 'Location', right_on = 'ClipName')
            print('Epoch: ' + str(i))
            acc_dt = out_dt.groupby('AnalysisID').agg({'Match':'mean','Location':'count'})
            print(acc_dt)
            acc_dt = out_dt.groupby(['AnalysisID','ProjectID']).agg({'Match':'mean','Location':'count'})

            #print(out_dt[out_dt.Probability > 0.8].groupby('AnalysisID').agg({'Match':'mean','Location':'count'}))
            acc_dt.to_csv(self.resultsDirectory + 'epoch_' + str(i) + '_accuracy.csv')

            scheduler.step(validation_loss)
            #if i % 5 == 0:
            #    _ = self.val_epoch(i, self.val_loader, model, criterion, self.val_logger)

    def make_predictions(self, trained_model, n_classes, output_file):
            
        model = resnet18(
                num_classes=n_classes,
                sample_size=self.sample_size,
                sample_duration=self.sample_duration)


        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        parameters = model.parameters()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
                                  

        checkpoint = torch.load(trained_model)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        _,confusion_matrix,confidence_matrix, results_df = self.val_epoch(0, self.val_loader, model, criterion, self.val_logger)
        with open(self.source_json_file,'r') as input_f:
            source_json = json.load(input_f)
        confidence_matrix.columns = source_json['labels']
        confidence_matrix['predicted_label'] = confidence_matrix.idxmax(axis="columns")
        confidence_matrix.to_csv(output_file)
        # pdb.set_trace()
        return
    
    def train_epoch(self, epoch, data_loader, model, criterion, optimizer, epoch_logger, checkpoint):
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

            """batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            })"""

            """print('Epoch: [{0}][{1}/{2}]\t'
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
                acc=accuracies))"""
        epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

        if epoch % checkpoint == 0:
            save_file_path = os.path.join(self.resultsDirectory,
                                          'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)

    def val_epoch(self, epoch, data_loader, model, criterion, logger):
        print('validation at epoch {}'.format(epoch))

        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        confusion_matrix = np.zeros((self.n_classes,self.n_classes))
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

                

                """print('Epoch: [{0}][{1}/{2}]\t'
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
                    acc=accuracies))"""
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
        
    
