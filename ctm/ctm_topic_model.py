from utils.data_loader import TwitterDataset
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.models.ctm import CTM
from config import logger

TRAIN_SIZE = 100000
BATCH_SIZE = 10000
NUM_TOPICS = 100
MINI_BATCH_SIZE = 512
HIDDEN_UNITS = (512, 256)

class TopicModel():
    def __init__(self, train_size: int = TRAIN_SIZE, n_topics: int = NUM_TOPICS, batch_size: int = BATCH_SIZE, shuffle=True):
        data_loader = TwitterDataset("TRAIN", train_size, batch_size=batch_size, shuffle=shuffle)
        data_loader.init()
        self.data_loader = data_loader
        self.ctm = CTM(input_size=len(self.data_loader.vocab), bert_input_size=512, num_epochs=20,
                       batch_size=MINI_BATCH_SIZE, inference_type="contextual", n_components=n_topics,
                       reduce_on_plateau=True,
                       lr=1e-4,
                       hidden_sizes=HIDDEN_UNITS,
                       num_data_loader_workers=0)

    def train(self):
        for i in range(self.data_loader.n_batches):
            logger.info(f"Starting Batch {i+1}")
            self.data_loader.encoded_next_batch()
            training_dataset = CTMDataset(self.data_loader.bow, self.data_loader.bert_embeddings, self.data_loader.idx2token)
            self.ctm.fit(training_dataset)
            logger.info("\n---------------------------")
            logger.info("--------Topics---------")
            topics = self.ctm.get_topic_lists(10)
            for t in topics:
                logger.info(t)
            logger.info("---------------------------\n")
        return self.ctm.get_topic_lists(10)

