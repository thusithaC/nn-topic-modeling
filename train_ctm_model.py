from topic_model import TopicModel
from config import logger

if __name__ == "__main__":
    topic_model = TopicModel(train_size=5000000, batch_size=40000)
    n_rounds = 5
    for r in range(n_rounds):
        logger.info(f"starting round : {r}")
        topic_model.train()
        topic_model.ctm.get_topic_lists(10)
        topic_model.data_loader.reset_training_data()
