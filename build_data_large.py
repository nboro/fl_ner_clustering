from model.config import Config
from model.data_utils import OldCoNLLDataset, CoNLLDataset, get_vocabs, get_full_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, get_full_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def main():
   
    # 1. get config and processing of words
    config = Config(load=False, dataset='full')

    #2. Get processing word generator
    processing_word = get_processing_word(lowercase=True)

    # 3. Generators

    ade = OldCoNLLDataset(config.filename_ade, processing_word)
    cadec = OldCoNLLDataset(config.filename_cadec, processing_word)

    

    # # 4. Build Word and Tag vocab
    vocab_words, vocab_tags_ade, vocab_tags_cadec = get_full_vocabs(ade=ade, cadec=cadec)
    vocab_glove = get_glove_vocab(config.filename_glove)

    # 5. Get a vocab set for words in both vocab_words and vocab_glove
    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # 6. Save vocab
    #sort words for consistency with other clients
    write_vocab(sorted(vocab), config.filename_full_words)
    # write_vocab(vocab, config.filename_full_words)
    write_vocab(vocab_tags_ade, 'data/ade/tags.txt')
    write_vocab(vocab_tags_cadec, 'data/cadec/tags.txt')

    # 7. Trim GloVe Vectors
    vocab = load_vocab(config.filename_full_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    ade = OldCoNLLDataset(config.filename_ade)
    cadec = OldCoNLLDataset(config.filename_cadec)

    # Build and save char vocab
    vocab_chars = get_full_char_vocab(ade, cadec)
    write_vocab(vocab_chars, config.filename_full_chars)


if __name__ == "__main__":
    main()
