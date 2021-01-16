import argparse, subprocess, datetime, os, pdb, sys



def main():
    
    parser = argparse.ArgumentParser(description='this script convert confidence scores to labels')

    parser.add_argument('--confidence_file',
                        type = str, 
                        default = '/data/home/llong35/temp/test_JAN_15_temp/prediction_confusion.csv',
                        required = False, 
                        help = 'confidence file')
    
    parser.add_argument('--prediction_file',
                        type = str, 
                        default = '/data/home/llong35/temp/test_JAN_15_temp/prediction_confusion_label.csv',
                        required = False, 
                        help = 'label_prediction')

    args = parser.parse_args()
    
    pdb.set_trace()
    with open(args.confidence_file,'r') as input, open(args.prediction_file,'w') as output:
        labels = input.readline().rstrip().split(',')[1:]
        output.write('Location,predicted_label\n')
        for line in input:
            tokens = line.rstrip().split(',')
            labels = tokens[1:]
            max_index = 0
            max_value = tokens[1]
            for i in range(len(tokens)-2):
                if tokens[i+2] > max_value:
                    max_index = i+1
                    max_value = tokens[i+2]
            output.write(tokens[0]+','+labels[max_index]+'\n')
            

if __name__ == "__main__":
    main()
    
