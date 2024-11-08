import os


from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, load=True, dataset=None, fed=False):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None
            dataset: ade or cadec 
        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)
        self.dataset = dataset
        self.fed = fed

        self.filename_dev = "data/{}/val.tsv".format(dataset)
        self.filename_test = "data/{}/test.tsv".format(dataset)
        self.filename_train = "data/{}/train.tsv".format(dataset)
        self.filename_cadec = "data/cadec/cadec.tsv"
        self.filename_ade = "data/ade/ade.tsv"
        self.filename_full_dataset = "data/{}/{}.tsv".format(dataset, dataset)

        self.filename_words = "data/{}/words.txt".format(dataset)
        self.filename_full_words = "data/full_words.txt"
        self.filename_tags = "data/{}/tags.txt".format(dataset)
        self.filename_chars = "data/{}/chars.txt".format(dataset)
        self.filename_full_chars = "data/full_chars.txt"
        if(dataset=='cadec'):
            self.batch_size = 32
        else:
            self.batch_size = 32
            # self.batch_size = 16

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """

        # 1. vocabulary
        if(self.fed==True):
            self.vocab_words = load_vocab(self.filename_full_words)
            self.vocab_chars = load_vocab(self.filename_full_chars)
        else:
            self.vocab_words = load_vocab(self.filename_words)
            self.vocab_chars = load_vocab(self.filename_chars)
        
        self.vocab_tags  = load_vocab(self.filename_tags)
       
        # print(len(self.vocab_words))
        

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
#     filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    filename_glove = "data/glove.6B.300d.txt"
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    


    # training
    train_embeddings = False
    nepochs          = 8
    dropout          = 0.5
    #moved batch size param to class init
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    epoch_drop       = 1 # Step Decay: per # epochs to apply lr_decay
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # ner_model_path = "saves/ner_{}e_glove".format(nepochs)
    ner_model_path = "saves/ner_fed"

    # elmo config
    use_elmo = True
    dim_elmo = 1024

    use_crf = True
    use_chars = True
