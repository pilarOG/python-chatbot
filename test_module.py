# this script allows to use a csv file to test your model
# and it will give you an accuracy report of the performance of your model
# over this labelled test set

from train_module import * # TODO: I know this is wrong!
import pickle

with open('chatbot-model.pckl', 'rb') as handle:
    pipeline = pickle.load(handle)

def check_classes(classes):
    #if in the test data there are classes that don't exist in the model
    unknown = []
    [unknown.append(n) for n in set(classes) if n not in set(pipeline.classes_)]
    if unknown != []:
        raise Exception('There is a class in your test data that does not exist in the model: '+' '.join(unknown))

def predict(test_data):
    # TODO: load any model
    y_pred = pipeline.predict(test_data)
    # this test does not allow a threshold
    correct = 0
    print '\'example\'\t\'class\'\t\'prediction\''
    for sample, category, prediction in zip(test_data['example'], test_data['class'], y_pred):
        print sample, '\t', category, '\t', prediction
        if category == prediction:
            correct += 1
    print 'accuracy: '+str(correct * 100/len(test_data))+'%'

if __name__ == "__main__":
    pd_data = data_processing(sys.argv[1])
    pd_feats = feature_extraction(pd_data['example'])
    pd_input = pd.concat([pd_data['class'], pd_data['example'], pd_feats], axis=1)
    check_classes(pd_data['class'])
    predict(pd_input)
