import flwr as fl
import sys
from typing import Dict, List, Tuple
import numpy as np
from collections import OrderedDict
import torch

from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
from model.cluster_datasets import cluster_dataset


class FedNERClient(fl.client.NumPyClient):

    def __init__(self, model, train, dev, test, dataset, fed=False) -> None:
        self.model = model
        self.train_data = train
        self.dev_data = dev
        self.test_data = test
        self.dataset = dataset
        self.fed = fed
        self.fed_layer_params = ['emb.weight', 'char_embeddings.weight', 'char_lstm.weight_ih_l0', 'char_lstm.bias_ih_l0', 
        'char_lstm.weight_hh_l0', 'char_lstm.bias_hh_l0', 'char_lstm.weight_ih_l0_reverse', 'char_lstm.bias_ih_l0_reverse',
         'char_lstm.weight_hh_l0_reverse', 'char_lstm.bias_hh_l0_reverse', 'char_lstm.l0.ih.weight', 
         'char_lstm.l0.ih.bias', 'char_lstm.l0.hh.weight', 'char_lstm.l0.hh.bias', 'char_lstm.l0_reverse.ih.weight', 
         'char_lstm.l0_reverse.ih.bias', 'char_lstm.l0_reverse.hh.weight', 'char_lstm.l0_reverse.hh.bias']

    def get_parameters(self) -> List[np.ndarray]:
        self.model.train()
        
        #get parameteres of of all embedding layers
        return [val.cpu().numpy() for key, val in self.model.state_dict().items() if key in self.fed_layer_params]
    
    def set_parameters(self, new_parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        # print("shapes")
        # for p in new_parameters:
        #     print(p.shape)
        # print(len(self.fed_layer_params))
        self.model.train()
        old_params = [val.cpu().numpy() for _, val in self.model.state_dict().items() ]
        # print(len(old_params))
        
        #update parameters of first layers with new 
        updated_params = new_parameters + old_params[len(new_parameters) :]
        print(len(updated_params))
        print('testing if weights get updated correctly')
        print(f'new emb layer mean weight: {new_parameters[0].mean()}')
        print(f'old emb layer mean weight: {old_params[0].mean()}')
        params_dict = zip(self.model.state_dict().keys(), updated_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        
        print("I AM FITTING")
        self.set_parameters(parameters)
        ner_config = Config(dataset = self.dataset, fed = self.fed)
        #decide when to save the model -> save on last round 
        print(config)
        save = config['current_round'] == config['rounds'] 
        print(save)
        learn = NERLearner(ner_config, self.model)
        learn.fit(self.train_data, epochs= config['local_epochs'], save=save, cluster = config['cluster'] if config['cluster'] else '') #add dev data if you want dev metrics per epoch
        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        ner_config = Config(dataset=self.dataset, fed = self.fed)
        learn = NERLearner(ner_config, self.model)
        f1, loss, accuracy = learn.evaluate(self.test_data)
        return float(loss), len(self.test_data), {"accuracy": float(accuracy)}

def main(argv) -> None:

    if (len(argv)==0):
        print("add command line argument for dataset to train on")
        return
    dataset = argv[0]
    cluster = argv[1] if len(argv)>1 else None

    
    # if config.use_elmo: config.processing_word = None
    

    config = Config(dataset=dataset, fed = True)
    full_dataset = CoNLLDataset(config.filename_full_dataset, config.processing_word,
            config.processing_tag, config.max_iter)

    #load dataset with only own vocabs and cluster
    if cluster: 
        cluster_indices = cluster_dataset(full_dataset, dataset, int(cluster))
        #get cluster dataset based on indices but with full vocabs
        cluster_data = torch.utils.data.Subset(full_dataset, cluster_indices)
        print('cluster data is : ')
        print(len(cluster_indices))
    else:
        cluster_data = full_dataset    

    config = Config(dataset=dataset, fed = True)
    model = NERModel(config)    
    N = len(cluster_data)
    n_train = int(0.8*N)
    # n_dev = int(0.1*N)
    n_test = N - n_train # - n_dev

    # train, dev, test = torch.utils.data.random_split(full_dataset, [n_train, n_dev, n_test])
    train, test = torch.utils.data.random_split(cluster_data, [n_train, n_test], generator=torch.Generator().manual_seed(1))

    print(len(train))
    print(len(test))
    ip = "localhost:500{}".format(cluster if cluster else 5)
    print('Running at: {}'.format(ip))

    # client = FedNERClient(model=model, train=train, dev=None, test=test, dataset=dataset, fed=True)
    # fl.client.start_numpy_client(ip, client)

if __name__ == "__main__":
    main(sys.argv[1:])