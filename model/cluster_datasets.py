import numpy as np
import torch
import sys
from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
import pickle
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering

#average representations (embeddings) of words in sentence to represent sentence as 1 vecor
def vectorizer(sentence):
    acc = np.zeros(len(sentence[0]))
    for i, word in enumerate(sentence):
        acc = np.add(acc, word)
    return acc/len(sentence)

def cluster_dataset(data, dataset=None, cluster=None):
   
    # if (len(argv)==0):
    #     print("add command line argument for dataset to train on")
    #     return
    # dataset = argv[0]

    # create instance of config
    config = Config(dataset=dataset, fed=True)
    # if config.use_elmo: config.processing_word = None

    #build model
    model = NERModel(config)

    learn = NERLearner(config, model)
    learn.load('saves/ner_fed_{}'.format(dataset))

    # data = CoNLLDataset(config.filename_train, config.processing_word,
    #                          config.processing_tag, config.max_iter)
    print('before cluster data is:') 
    print(len(data)) 

    print('Calculating embeddings...')                  
    result = learn.evaluate(data, extracting = True)
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
    print('vectorized sentences shsape:')
    print(vectorized_sentences.shape)

    # load the model from disk
    cluster_model = pickle.load(open('data/og_kmeans_cl_model.sav', 'rb'))
    common_scaler = pickle.load(open('data/og_scaler.sav', 'rb'))

    # cluster_model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    scaled = preprocessing.StandardScaler().fit_transform(vectorized_sentences)
    scaled = common_scaler.transform(scaled)

    preds = cluster_model.predict(scaled)

    unique, counts = np.unique(preds, return_counts=True)

    print(np.asarray((unique, counts)).T)
    
    cluster_indices = np.where(preds == cluster)
    print("samples in cluster: {}".format(len(cluster_indices[0])))
    return cluster_indices[0]
