import sys

from model.config import Config
from model.data_utils import OldCoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def main(argv):
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...
        command line: data set name argument to build vocabs for selected dataset
    """

    if (len(argv)==0):
        print("add command line argument for dataset")
        return
    dataset = argv[0]
    # 1. get config and processing of words
    config = Config(load=False, dataset=dataset)

    #2. Get processing word generator
    processing_word = get_processing_word(lowercase=True)

    # 3. Generators
    dev   = OldCoNLLDataset(config.filename_dev, processing_word)
    test  = OldCoNLLDataset(config.filename_test, processing_word)
    train = OldCoNLLDataset(config.filename_train, processing_word)


    # 4. Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    # 5. Get a vocab set for words in both vocab_words and vocab_glove
    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # 6. Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # 7. Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = OldCoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main(sys.argv[1:])
