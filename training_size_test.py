import os
import numpy as np
import pandas as pd
import sys
from subprocess import call
import json
import pdb

def convert_csv_to_dict(csv_path, subset):
    try:
        data = pd.read_csv(csv_path, delimiter=' ', header=None)
    except:
        print('Warning: no {}, check data'.format(csv_path))
        return {}
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        slash_rows = data.ix[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1]
        
        keys.append(basename)
        key_labels.append(class_name)
        
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}
    
    return database

def split_train_validation_test(result_path,video_path,train_ratio,validation_ratio):
    train_list_csv = os.path.join(result_path,'train_list.csv')
    val_list_csv = os.path.join(result_path,'val_list.csv')
    test_list_csv = os.path.join(result_path,'test_list.csv')
    dst_json_path = os.path.join(result_path,'cichlids.json')
    with open(train_list_csv,'w') as train_output, open(val_list_csv,'w') as val_output,open(test_list_csv,'w') as test_output:
        for folder in os.listdir(video_path):
            folder_path = os.path.join(video_path,folder)
            if not os.path.isdir(folder_path):
                    continue
            for file in os.listdir(folder_path):
                if not file.endswith('.mp4'):
                    continue
                output_string = folder+'/'+file.split('.')[0]+'\n'
                random_number = np.random.uniform()
                if  random_number < train_ratio:
                    train_output.write(output_string)
                elif random_number < train_ratio+validation_ratio:
                    val_output.write(output_string)
                else:
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
    
    
    
def prepare_data_directories(excel_file,master_directory,data_folder):
    # for each training, create a new folder for training results watching
    df = pd.read_excel(excel_file)
    for index, row in df.iterrows():
        train_ratio = row['training_precentage']
        val_ratio = row['validation_percentage']
        test_ratio = row['testing_percentage']
        # create a new directory
        directory = os.path.join(master_directory,'{}-{}-{}'.format(train_ratio,val_ratio,test_ratio))
        call(['mkdir',directory])
        call(['ln','-s',data_folder,directory+'/'])
        split_train_validation_test(directory,data_folder,train_ratio,val_ratio)

def main():
    
    excel_file = '/data/home/llong35/files_for_3D_resnet/training_percentage.xlsx'
    master_directory = '/data/home/llong35/data/transfer_test'
    data_folder = '/data/home/llong35/data/annotated_videos'
    prepare_data_directories(excel_file,master_directory,data_folder)

if __name__ == '__main__':
    main()

