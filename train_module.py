# This script takes as input a path to a csv file with the format 'example' and 'class'
# to train an sklearn text classifier, which is the output of it.
# This chatbot is designed for Spanish and therefore, it uses utf-8 encoding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from pattern.es import parse
import pandas as pd
import pickle
import sys

# TODO: add help
# TODO: add log

# DATA PROCESSING
def data_processing(path_csv):
    data = pd.read_csv(path_csv, header = 0, delimiter = ",", encoding = 'utf-8')
    # check for correct headers
    if 'example' not in data or 'class' not in data:
        raise Exception('Data headers are wrong')
    # check for empty values
    if data.isnull().values.any():
        raise Exception('There are missing values')
    # check that the number of classes is greater than one
    if len(set(data['class'])) == 1:
        raise Exception('You must have more than one class')
    # TODO: check balancing of examples per class
    return data

# FEATURE EXTRACTION
# this should be your most important and customizable function
# here we will just use the pattern library to work with lemmas
def feature_extraction(pd_df):
    all_lemmas, all_tokens = [], []
    for row in pd_df:
        lemmas, tokens = '', ''
        parsed = parse(row, lemmata=True)
        for token in parsed.split():
            for item in token:
                lemmas += item[-1]
                tokens += item[0]
        all_lemmas.append(lemmas)
        all_tokens.append(tokens)
    # create data frame with features
    return pd.DataFrame(data = {"lemmas":all_lemmas,
                                "tokens":all_tokens})

# FEATURE ENCODING
# sklearn classes to encode the features to use as input for the model
class VectTokens(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = CountVectorizer()
    def fit(self, data, y=None):
        self.model.fit(data['tokens'])
        return self
    def transform(self, data):
        # print this & .shape to see matrix values
        return self.model.transform(data['tokens'])

class TokensTfIdf(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = TfidfVectorizer()
    def fit(self, data, y=None):
        self.model.fit(data['tokens'])
        return self
    def transform(self, data):
        # print this & .shape to see matrix values
        return self.model.transform(data['tokens'])

class VectLemmas(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = CountVectorizer()  ### TODO: OPTIMIZE PARAMETERS
    def fit(self, data, y=None):
        self.model.fit(data['lemmas'])
        return self
    def transform(self, data):
        # print this & .shape to see matrix values
        return self.model.transform(data['lemmas'])

class LemmasTfIdf(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = TfidfVectorizer()   ### TODO: OPTIMIZE PARAMETERS
    def fit(self, data, y=None):
        self.model.fit(data['lemmas'])
        return self
    def transform(self, data):
        # print this & .shape to see matrix values
        return self.model.transform(data['lemmas'])

# TRAINING
def train_model(training_data):
    # Encoding of the features and feature union
    features = []
    features.append(('token_tfidf', TokensTfIdf()))
    features.append(('token_vect', VectTokens()))
    features.append(('lemma_tfidf', LemmasTfIdf()))
    features.append(('lemma_vect', VectLemmas()))
    # you could add weights at this point to each feature
    all_features = FeatureUnion(features)

    # define model and hyperparameters #TODO: add the option to use other models
    Classifier = SGDClassifier(loss='log', penalty='l2',
                              alpha=1e-3, n_iter=5, random_state=42)

    # create pipeline (which is what we will actually output as a pickle)
    pipeline = Pipeline([('all', all_features),
                         ('clf', Classifier),])

    # train and save (output) the pipeline
    pipeline.fit(training_data, training_data['class'])
    with open('chatbot-model.pckl', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

# MAIN - this must be another function so pickle will not have issues finding the transforms
def train(csv_file):
    pd_data = data_processing(csv_file)
    pd_feats = feature_extraction(pd_data['example'])
    pd_input = pd.concat([pd_data['class'], pd_feats], axis=1)
    train_model(pd_input)

if __name__ == "__main__":
    train(sys.argv[1]) # TODO: add error here
