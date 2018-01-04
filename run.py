# use this to test in real time an example and get an answer
# turn this into a service and plug it to a front-end
# the input is a string of text and so the output
# you could get the probability of the answer and set thresholds if you want

from train_module import *
from test_module import *
import sys

with open('chatbot-model.pckl', 'rb') as handle:
    pipeline = pickle.load(handle)
with open('chatbot-dialog.pckl', 'rb') as handle:
    dialog = pickle.load(handle)

def preprocess(input):
    # check that is string or unicode, and not empty
    if input == '' or type(input) != str and type(input) != unicode:
        raise Exception('Input format not valid')
    pd_data = pd.DataFrame(data = {"example":input})
    pd_feats = feature_extraction(pd_data['example'])
    return pd.concat([pd_data['class'], pd_feats], axis=1)

def answer(feats):
    y_pred = pipeline.predict_proba(test_data)
    print y_pred


if __name__ == __main__:
    input = sys.argv[1]
    feats = preprocess(input)
