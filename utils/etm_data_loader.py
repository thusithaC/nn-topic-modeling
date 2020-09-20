from utils.mongo_helper import MongoHelper
from typing import Text
from config import MONGODB_HOST_PORT, MONGO_DB_NAME, MONGODB_TWEETS_COLLECTION, logger
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
import itertools
from functools import partial
import random
from torchtext.vocab import FastText

MIN_WORD_FREQUENCY = 20
OOV_SYMBOL = "__OOV__"

def get_bag_of_words(data, min_length):

    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]
    return np.array(vect)


def encode_text(label_encoder, sentences):
    encoded_sentences = []
    for t in tqdm(sentences):
        tokens = [w if (w in label_encoder.classes_) else OOV_SYMBOL for w in t.split()]
        t_e = label_encoder.transform(tokens)
        encoded_sentences.append(t_e)
    return encoded_sentences


class TwitterDataset():
    def __init__(self, mode: Text, size: int, batch_size: int = 50000, shuffle=True, workers=8):
        self.tweets = None
        self.mode = mode
        self.data_len = size
        self.batch_size = batch_size
        self.vocab_dict = {}
        self.vocab = []
        self.index_dd = None
        self.idx2token = None
        self.training_bow = None
        self.bow = None
        self.n_batches = None
        self.batch_generator = None
        self.shuffle = shuffle,
        self.workers = workers

    def init(self):
        if self.mode == "TRAIN":
            filters = {"lang": "en"}
        else:
            filters = {}
        db = MongoHelper(MONGODB_HOST_PORT, db_name=MONGO_DB_NAME)
        logger.info("loading data")
        data = db.select(MONGODB_TWEETS_COLLECTION, filters, limit=self.data_len)
        logger.info(f"loaded data {len(data)} items")
        processed_tweets = [d.get('processed') for d in data if d.get("processed") is not None and d.get("processed") != ""]
        if self.shuffle:
            random.shuffle(processed_tweets)
        self.tweets = processed_tweets
        self.n_batches = int(np.ceil(len(processed_tweets)/self.batch_size))

        concatenate_text = ' '.join(processed_tweets)
        all_words = list(concatenate_text.split())
        word_count = {}
        for w in all_words:
            count = word_count.get(w, 0)
            count += 1
            word_count[w] = count
        self.vocab = [w for w, i in word_count.items() if i >= MIN_WORD_FREQUENCY and len(w) > 1 ] + [OOV_SYMBOL]
        logger.info(f"Vocab size {len(self.vocab)}")

        label_encoder = LabelEncoder()
        logger.info("creating label encoding")
        label_encoder.fit(self.vocab)
        self.label_encoder = label_encoder

        logger.info("creating label encoding complete")
        transformed_labels = label_encoder.transform(self.vocab)
        for vocab, label in zip(self.vocab, transformed_labels):
            self.vocab_dict[vocab] = label

        self.idx2token = {v: k for (k, v) in self.vocab_dict.items()}
        logger.info("encoding all data")
        self.batch_generator = self.get_batch()
        logger.info("Initializing complete")

    def get_batch(self):
        for i in range(0, self.n_batches):
            yield self.tweets[i*self.batch_size: (i+1)*self.batch_size]

    def encode_next_batch(self):
        processed_tweets = self.batch_generator.__next__()
        encode_tweets = partial(encode_text, self.label_encoder)
        logger.info(f"Obtained next batch of tweets of length {len(processed_tweets)}")
        logger.info("Creating BOW encoding")
        if self.workers > 1:
            n_proc = self.workers
            with Pool(n_proc) as pool:
                tweets_chunks = np.array_split(processed_tweets, n_proc)
                encoded_chunks = pool.map(encode_tweets, tweets_chunks)
                encoded = list(itertools.chain(*encoded_chunks))
        else:
            encoded = encode_tweets(processed_tweets)
        self.index_dd = np.array(encoded)
        self.bow = get_bag_of_words(self.index_dd, len(self.vocab))
        logger.info("Creating BOW encoding complete")
        return self.bow

    def reset_training_data(self):
        """ reshuffle and reset the data"""
        logger.info("resetting training data")
        if self.shuffle:
            random.shuffle(self.tweets)
        self.batch_generator = self.get_batch()

    def inverse_transform_token(self, token_id: int):
        return self.label_encoder.inverse_transform([token_id])[0]

    def transform_token(self, token: str):
        try:
            return self.label_encoder.transform([token])[0]
        except:
            return self.label_encoder.transform([OOV_SYMBOL])[0]

    def create_embedding_matrix(self):
        """ currently only supports fasttext.
        Returns the weight matrix for the current vocab"""
        import torch
        import torch.nn as nn
        target_vocab = self.label_encoder.classes_
        embedding = FastText('en')
        emb_dim = int(embedding.dim)
        matrix_len = len(target_vocab)
        weights_matrix = np.zeros((matrix_len, emb_dim))
        words_found = 0
        for i, word in enumerate(target_vocab):
            try:
                weights_matrix[i] = embedding[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

        weights_matrix = torch.from_numpy(weights_matrix)

        return weights_matrix