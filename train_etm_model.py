from etm.embedded_topic_model import EtmModel
from utils.etm_data_loader import TwitterDataset
from etm.etm_data import ETMDataset
from config import logger

N_SAMPLES = 1000000
BATCH_SIZE = 20000
if __name__ == "__main__":

    raw_data = TwitterDataset(mode="TRAIN", size=N_SAMPLES, batch_size=BATCH_SIZE)
    raw_data.init()
    logger.info("obtained data")
    vocab_size = len(raw_data.vocab)
    emb_weights = raw_data.create_embedding_matrix()

    topic_model = EtmModel(vocab_size=vocab_size, batch_size=512, embeddings=emb_weights, train_embeddings=False)
    n_batches = int(N_SAMPLES/BATCH_SIZE)
    for i in range(n_batches):
        X = raw_data.encode_next_batch()
        etm_data = ETMDataset(X)
        topic_model.train(etm_data, epochs=10)
        logger.info(f"Outer batch {i+1}/{n_batches} completed")

        text = "president has no clue what we can do to fight the coronavirus"
        topic_model.visualize(text, raw_data)