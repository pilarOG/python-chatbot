# This script takes a csv to create a dictionary to obtain one or more answers
# of the chatbot for each class
# use '/' to separate multiple answers in the field 'answer'
# dict is saved as pickle dict

# the idea to separete this from main training is that if you want to change
# a comma in an answer you don't have to train all over again :)

import pandas as pd
import pickle
import sys

def open_data(path_csv):
    data = pd.read_csv(path_csv, header = 0, delimiter = ",", encoding = 'utf-8')
    # check for correct headers
    if 'answer' not in data or 'class' not in data:
        raise Exception('Data headers are wrong')
    if data.isnull().values.any():
        raise Exception('There are missing values')
    return data

def create_dialog(data):
    dialog = {}
    for n in range(0, len(data['class'])):
        dialog[data['class'][n]] = data['answer'][n].split('/')
    with open('chatbot-dialog.pckl', 'wb') as handle:
        pickle.dump(dialog, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print dialog

if __name__ == "__main__":
    data = open_data(sys.argv[1]) # TODO: add error here
    create_dialog(data)
