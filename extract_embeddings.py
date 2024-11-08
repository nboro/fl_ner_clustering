from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
import sys
import numpy as np
import pandas as pd
import tqdm


#average representations (embeddings) of words in sentence to represent sentence as 1 vecor
def vectorizer(sentence):
    acc = np.zeros(len(sentence[0]))
    for i, word in enumerate(sentence):
        acc = np.add(acc, word)
    return acc/len(sentence)


def main(argv):

    if (len(argv)==0):
        print("add command line argument for dataset to train on")
        return
    dataset = argv[0]

    # create instance of config
    config = Config(dataset=dataset, fed=True)
    # if config.use_elmo: config.processing_word = None

    #build model
    model = NERModel(config)

    learn = NERLearner(config, model)
    learn.load('saves/ner_fed_{}'.format(dataset))
  

    small = CoNLLDataset(config.filename_train, config.processing_word,
                             config.processing_tag, config.max_iter)

    print(len(small))
    #list of batches of embeddings in np array form
    result = learn.evaluate(small, extracting = True)
    # print(result[0])

    counter = 0

    #count sentences of all all batches except last + length of last smaller batch
    n_sentences = (len(result)-1)*config.batch_size + len(result[len(result)-1]) 
    vectorized_sentences = np.zeros((n_sentences, result[0].shape[2]))
    # vectorized_sentences = np.empty((0,1524), float)
    for batch in result:
        for i, sentence in enumerate(batch):
            vectorized_sentences[counter] = vectorizer(sentence)
            # vectorized_sentences = np.append(vectorized_sentences, vectorizer(sentence), axis=0)
            counter+=1

    print(vectorized_sentences.shape)
    print(counter)
    # print(vectorized_sentences[len(vectorized_sentences)-1])
    df = pd.DataFrame(vectorized_sentences)
    df.to_csv('data/{}/extracted_embs.csv'.format(dataset), header=False, index=False)


if __name__== "__main__":
    main(sys.argv[1:])