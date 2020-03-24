import matplotlib.pyplot as plt
import os
import numpy as np

import pdb

def interpret_log(log_file):
    with open(log_file,'r') as input:
        input.readline()
        epochs = []
        accs = []
        for line in input:
            epoch,loss,acc = line.split()[:3]
            epoch = int(epoch)
            acc = float(acc)
            epochs.append(epoch)
            accs.append(acc)
    return epochs,accs

def plot_accuracies(results_folder):
    training_file = os.path.join(results_folder,'train.log')
    val_file = os.path.join(results_folder,'val.log')
    test_file = os.path.join(results_folder,'test.log')
    output_file = os.path.join(results_folder,'accuracy_plot.png')
    
    train_epochs,training_acc = interpret_log(training_file)
    val_epochs,val_acc = interpret_log(val_file)
    test_epochs,test_acc = interpret_log(test_file)
    
    fig = plt.figure()
    plt.plot(train_epochs,training_acc,label='training accuracy')
    plt.plot(val_epochs,val_acc,label='validation accuracy')
    plt.plot(test_epochs,test_acc,label='test accuracy')
    plt.legend()
#     pdb.set_trace()
    plt.xticks(np.arange(0, train_epochs[-1]+1, 5))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.savefig(output_file)
    
    
def main():
    results_folder = '/data/home/llong35/data/transfer_test/animal_split/MC6_5,MCxCVF1_12b_1,MCxCVF1_12a_1,MC16_2,TI2_4,CV10_3/results'
    plot_accuracies(results_folder)

if __name__ == '__main__':
    main()