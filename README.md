# PyTorch implmenetation of federated NER model with in-client clustering



## Usage
1.	**Requirements**:  
    -	Packages: Pytorch, AllenNLP, sklearn, (AllenNLP requires linux), GPU strongly recommended
    -	Data: Train, valid and test dataset files as well as complete dataset file in CoNLL 2003 NER format place in path 'data/{dataset_name}'
    -	Glove 300B embeddings 
    
2.	**Configure Settings**:  
    -	Change settings in model/config.py  
    -	Main settings to change: File directories, model hyperparameters etc.  
    
3.	**Execution**
    - Build single vocabularies with build_data.py
    - Build large vocabularies with build_data_large.py
    - If using clustering, create embedded data with extract_embeddings.py and cluster with cluster_client.ipynb
    - Run server and clients with required arguments for data and cluster number
    - For solo training use train.py (change model path in config)
    - For testing on test data use testing.py

    Code base from [here](https://github.com/yongyuwen/PyTorch-Elmo-BiLSTMCRF)
