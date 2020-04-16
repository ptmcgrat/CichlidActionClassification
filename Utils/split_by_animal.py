import os
import numpy as np
import pandas as pd
import sys
from subprocess import call
import json
import pdb

from training_size_test import convert_csv_to_dict

def create_random_spliting_train_test(annotation_file,
                                        master_dir,
                                        data_folder,
                                        n_training=6,
                                        split_ratio = 0.8,
                                        training_sample_size = 9500,
                                        val_sample_size = 2000,
                                        test_sample_size = -1,
                                        test_in_train = 800):
    animals_list = ['MC16_2', 'MC6_5', 'MCxCVF1_12a_1', 'MCxCVF1_12b_1', 'TI2_4', 'TI3_3', 'CV10_3']
    training = np.sort(np.random.choice(animals_list, n_training, replace=False))
    training = [ 'MC16_2', 'MC6_5', 'MCxCVF1_12a_1', 'MCxCVF1_12b_1', 'TI2_4', 'CV10_3']
    result_dir = os.path.join(master_dir,','.join(training)+'testintrain'+str(test_in_train))
    if os.path.isdir(result_dir):
        return
    else:
        call(['mkdir',result_dir])
        call(['ln','-s',data_folder,result_dir+'/'])
    train_list_csv = os.path.join(result_dir,'train_list.csv')
    val_list_csv = os.path.join(result_dir,'val_list.csv')
    test_list_csv = os.path.join(result_dir,'test_list.csv')
    dst_json_path = os.path.join(result_dir,'cichlids.json')
    
    
    annotateData = pd.read_csv(annotation_file, sep = ',', header = 0)

    i = 0
    train_list = []
    val_list = []
    test_list = []
    
    for index,row in annotateData.iterrows():
        output_string = row['Label']+'/'+row['Location']+'\n'
        animal = row['MeanID'].split(':')[0]
        #first determine if this is train/validation or test
        if animal not in training:
            test_list.append(output_string)
            continue
        #if train/validation, determine if this go to train or validation
        if np.random.uniform() < split_ratio:
            train_list.append(output_string)
                
        else:
            val_list.append(output_string)
                
    with open(train_list_csv,'w') as train_output:
        if training_sample_size > len(train_list):
            print('not enough training data to sample')
            raise
        if training_sample_size != -1:
            train_list = np.random.choice(train_list, training_sample_size, replace=False)
            test_in_train_list = np.random.choice(test_list, test_in_train, replace=False)
            for output_string in train_list:
                train_output.write(output_string)
            for output_string in test_in_train_list:
                train_output.write(output_string)
            
    with open(val_list_csv,'w') as val_output:
        if val_sample_size > len(val_list):
            print('not enough validation data to sample')
            raise
        if val_sample_size != -1:
            val_list = np.random.choice(val_list, val_sample_size, replace=False)
            for output_string in val_list:
                val_output.write(output_string)
    with open(test_list_csv,'w') as test_output:
        # if test_sample_size > len(test_list):
#             print('not enough test data to sample')
#             raise
#         if test_sample_size != -1:
#             test_list = np.random.choice(test_list, test_sample_size, replace=False)
        for output_string in test_list:
            temp = set(test_in_train_list)
            if output_string not in temp:
                test_output.write(output_string)
            
    
    train_database = convert_csv_to_dict(train_list_csv, 'training')
    val_database = convert_csv_to_dict(val_list_csv, 'validation')
    test_database = convert_csv_to_dict(test_list_csv, 'test')
    dst_data = {}
    dst_data['labels'] = ['c', 'f', 'p', 't', 'b', 'm', 's', 'x', 'o', 'd']
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    dst_data['database'].update(test_database)
    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)
            
def prepare_animal_subset_directories(excel_file,master_directory,data_folder):
    # for each training, create a new folder for training results watching
    
        # create a new directory
        directory = os.path.join(master_directory,'{}-{}-{}'.format(train_ratio,val_ratio,test_ratio))
        call(['mkdir',directory])
        
        split_train_validation_test(directory,data_folder,train_ratio,val_ratio)
        
def main():
    annotation_file = '/data/home/llong35/patrick_code_test/modelAll_34/AnnotationFile.csv'
    master_dir = '/data/home/llong35/data/transfer_test/animal_split'
    data_folder = '/data/home/llong35/data/annoated_videos_jpgs'
    create_random_spliting_train_test(annotation_file,master_dir,data_folder,6,split_ratio = 0.8)
if __name__ == '__main__':
    main()