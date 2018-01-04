# use this to test in real time an example and get an answer
# turn this into a service and plug it to a front-end
# the input is a string of text and so the output
# you could get the probability of the answer and set thresholds if you want

from train_module import *
from test_module import *
import sys
import random
import pickle
from flask import Flask, request, redirect, jsonify, make_response, current_app

app = Flask(__name__)

with open('chatbot-model.pckl', 'rb') as handle:
    pipeline = pickle.load(handle)
with open('chatbot-dialog.pckl', 'rb') as handle:
    dialog = pickle.load(handle)

def preprocess(input):
    # check that is string or unicode, and not empty
    if input == '' or type(input) != str and type(input) != unicode:
        raise Exception('Input format not valid')
    pd_data = pd.DataFrame(data = {"example":[input]})
    return feature_extraction(pd_data['example'])

@app.route('/prediction', methods=['POST'])
def bot_answer():
    input = request.json['input']
    answer = {'label':None,'input':None,'score':None,'answer':None}
    feats = preprocess(input)
    answer['input'] = input
    results = pipeline.predict_proba(feats)[0]
    resultsDict = dict(zip(pipeline.classes_, results))
    resultsRank = map(lambda x: x[0], sorted(zip(pipeline.classes_, results), key=lambda x: x[1], reverse=True))
    answer['label'] = resultsRank[0]
    answer['score'] = resultsDict[resultsRank[0]]
    answer['answer']= random.choice(dialog[answer['label']])
    print answer
    return jsonify(answer)

if __name__ == '__main__':
    app.run()
