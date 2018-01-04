# this script allows to use a csv file to test your model
# and it will give you an accuracy report of the performance of your model
# over this labelled test set

from train import * # TODO: I know this is wrong!
import pickle

def predict(test_data):
    # TODO: load any model
    with open('chatbot-model.pckl', 'rb') as handle:
        pipeline = pickle.load(handle)
    y_pred = pipeline.predict(test_data)
    # this test does not allow a threshold
    correct = 0
    for sample, category, prediction in zip(test_data['example'], test_data['class'], y_pred):
        print sample, category, prediction
        if category == prediction:
            correct += 1
    print 'accuracy: '+str(correct * 100/len(test_data))+'%'

if __name__ == "__main__":
    data = train.data_processing(sys.argv[1])
    predict(data)
