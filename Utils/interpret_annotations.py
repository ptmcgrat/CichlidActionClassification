import argparse
from collections import defaultdict
import pandas as pd
def interpret(annotation_file):
    location_to_label = {}
    location_to_meanid = {}
    meanID_to_label = defaultdict(list)
    df = pd.read_csv(annotation_file)
    for row in df.iterrows():
        meanID_to_location[row.MeanID].append(row.Location)
        location_to_label[row.Location] = row.Label
        location_to_meanid[row.location] = row.MeanID
    return location_to_label, location_to_meanid, meanID_to_label
        
    




def main():
    parser = argparse.ArgumentParser(description='interpret annotation file')
    parser.add_argument('--ML_labels', type = str, required = True, help = 'labels given to each ML video')
    args = parser.parse_args()
    return 

