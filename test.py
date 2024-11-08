""" Command Line Usage
Args:
    eval: Evaluate F1 Score and Accuracy on test set
    pred: Predict sentence.
    (optional): Sentence to predict on. If none given, predicts on "Peter Johnson lives in Los Angeles"

Example:
    > python test.py eval pred "Obama is from Hawaii"
"""

from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
import sys
from model.cluster_datasets import cluster_dataset
import torch


def main(argv):
    if (len(argv)==0):
        print("add command line argument for dataset to test on and cluster number if needed")
        return
    dataset = argv[0]
    cluster = argv[1] if len(argv)>1 else ''
    # create instance of config

    #use this config for federated training
    config = Config(dataset=dataset, fed=True)

    #use this config for solo training
    # config = Config(dataset=dataset)

    #build model
    model = NERModel(config)

    learn = NERLearner(config, model)
    #if loading the general model
    learn.load()

    #if loading cluster model
    # learn.load(cluster=cluster)

    # create datasets
    test = CoNLLDataset(config.filename_test, config.processing_word,
                            config.processing_tag, config.max_iter)

    if cluster: 
        cluster_indices = cluster_dataset(test, dataset, int(cluster))
        #get cluster dataset based on indices but with full vocabs
        cluster_data = torch.utils.data.Subset(test, cluster_indices)
        print('cluster data is : ')
        print(len(cluster_indices))
    else:
        cluster_data = test   
    
    f1, loss, acc, p, r = learn.evaluate(cluster_data)

    print('F1: {}, loss: {}, accuracy: {}, precision: {}, recall: {}'.format(f1, loss, acc, p, r))

    #old code for inference, may need debugging
    # if sys.argv[1] == "pred" or sys.argv[2] == "pred":
    #     try:
    #         sent = (sys.argv[2] if sys.argv[1] == "pred" else sys.argv[3])
    #     except IndexError:
    #         sent = "We report the cases of two patients who developed acute hepatitis after taking riluzole at the recommended dose (100 mg daily) for 7 and 4 weeks, respectively."

    #     print("Predicting sentence: ", sent)
    #     pred = learn.predict(sent)
    #     print(pred)



if __name__ == "__main__":
    main(sys.argv[1:])
