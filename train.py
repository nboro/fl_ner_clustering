from model.data_utils import CoNLLDataset
from model.config import Config
from model.ner_model import NERModel
from model.ner_learner import NERLearner
import sys 
# from torch.optim import Adam
# from model.ent_model import EntModel
# from model.ent_learner import EntLearner

# from opacus.validators import ModuleValidator


def main(argv):

    if (len(argv)==0):
        print("add command line argument for dataset to train on")
        return
    dataset = argv[0]

    # create instance of config
    config = Config(load=True, dataset=dataset)
    # if config.use_elmo:
    #     config.processing_word = None

    # build model
    model = NERModel(config)
    print(model.emb)

    # model.train()
    # errors = ModuleValidator.validate(model, strict=False)
    # print(errors)

    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter, config.use_crf)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter, config.use_crf)   

    print(len(train))
    # print(train[1])
    # train_loader = DataLoader(train, batch_size=64)
      

    learn = NERLearner(config, model)
    learn.fit(train, dev, epochs=8, save=True)
    # learn.fit(train)


if __name__ == "__main__":
    main(sys.argv[1:])
