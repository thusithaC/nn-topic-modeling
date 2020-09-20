from typing import Text, List, Dict, Union, Optional
from etm.etm_base import ETM
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from etm.etm_data import ETMDataset
import os
from config import logger
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EtmModel:
    """
    The EtmModel class will encapsulate the ETM NN model and the fit/transform functions.
    Params:
    activation: tanh, relu, softplus, rrelu, leakyrelu, elu, selu, glu
    """

    def __init__(self,
                 num_topics: int = 100,
                 vocab_size: int = 500,
                 t_hidden_size: int = 256,
                 rho_size: int = 500,
                 theta_act: str = "relu",
                 embeddings=None,
                 train_embeddings=True,
                 enc_drop: float = 0.25,
                 optimizer_name: str = "rmsprop",
                 lr: float = 1e-4,
                 wdecay: float = 1e-7,
                 bow_norm: bool = False,
                 clip: float = 0.0,
                 log_interval: int = 10,
                 batch_size: int = 512
                 ):
        self.optimizer_name = optimizer_name
        self.log_interval = log_interval
        self.lr = lr
        self.wdecay = wdecay
        self.bow_norm = bow_norm
        self.clip = clip
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.model = ETM(num_topics, vocab_size, t_hidden_size, rho_size, theta_act,
                         embeddings, train_embeddings, enc_drop).to(device)
        self.train_data = None
        self.optimizer: Optional[Optimizer] = None
        self.model_dir = None
        self.batch_size = batch_size
        self.num_data_loader_workers = 0
        self.init_model()

    def init_model(self):
        model = self.model
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wdecay)
            logger.info('Using Adam optimizer')
        elif self.optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=self.lr, weight_decay=self.wdecay)
            logger.info('Using RMSprop optimizer')
        else:
            logger.info('Defaulting to vanilla SGD')
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
        self.optimizer = optimizer

    def train(self, train_data: ETMDataset, epochs: int, save_dir: Text = None):
        self.train_data = train_data
        self.model_dir = save_dir
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers)
        for epoch in range(epochs):
            self._fit_epoch(train_loader, epoch)

    def _fit_epoch(self, loader, epoch):
        model = self.model
        optimizer = self.optimizer
        model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        n_batches = int(len(loader)/self.batch_size)
        for idx, batch_samples in enumerate(loader):
            optimizer.zero_grad()
            model.zero_grad()
            data_batch = batch_samples.to(device)
            sums = data_batch.sum(1).unsqueeze(1)
            if self.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            recon_loss, kld_theta = model(data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()

            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1

            if idx % self.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2)
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                logger.info('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, n_batches, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

        cur_loss = round(acc_loss / cnt, 2)
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        logger.info('*' * 100)
        logger.info('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        logger.info('*' * 100)

    def visualize(self, sentence, raw_data_set, show_emb=False, num_words=10):
        def nearest_neighbors(word, embeddings):
            vectors = embeddings.data.cpu().numpy()
            index = raw_data_set.transform_token(word)
            query = vectors[index]
            ranks = vectors.dot(query).squeeze()
            denom = query.T.dot(query).squeeze()
            denom = denom * np.sum(vectors ** 2, 1)
            denom = np.sqrt(denom)
            ranks = ranks / denom
            mostSimilar = []
            [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
            nearest_neighbors = mostSimilar[:20]
            nearest_neighbors = [raw_data_set.inverse_transform_token(comp) for comp in nearest_neighbors]
            return nearest_neighbors

        if not os.path.exists('./results'):
            os.makedirs('./results')
        m = self.model
        m.eval()
        queries = sentence.split()
        ## visualize topics using monte carlo
        with torch.no_grad():
            logger.info('#' * 100)
            logger.info('Visualize topics...')
            topics_words = []
            gammas = m.get_beta()
            for k in range(self.num_topics):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-num_words + 1:][::-1])
                topic_words = [raw_data_set.inverse_transform_token(a) for a in top_words]
                topics_words.append(' '.join(topic_words))
                logger.info('Topic {}: {}'.format(k, topic_words))

            if show_emb:
                ## visualize word embeddings by using V to get nearest neighbors
                logger.info('#' * 100)
                logger.info('Visualize word embeddings by using output embedding matrix')
                try:
                    embeddings = m.rho.weight  # Vocab_size x E
                except:
                    embeddings = m.rho  # Vocab_size x E
                neighbors = []
                for word in queries:
                    logger.info('word: {} .. neighbors: {}'.format(
                        word, nearest_neighbors(word, embeddings)))
                logger.info('#' * 100)