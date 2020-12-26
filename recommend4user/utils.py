"""
    Some handy functions for pytroch model training ...
"""
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam([{"params": [y[1] for y in network.named_parameters() if
                                                  ("attention" not in y[0]) and (y[1].requires_grad)]},
                                      {"params": [y[1] for y in network.named_parameters() if
                                                  ("attention" in y[0]) and (y[1].requires_grad)],
                                       "lr": params['adam_lr'] * params['lr_ratio']}], lr=params['adam_lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def load_friends_tweets(path):
    user_friends = {}
    user_tweets = {}


    cnt = 0
    df = pd.read_csv(path, header=0, dtype={0: np.int})
    for i, row in df.iterrows():
        try:
            friends = [int(i) for i in row[1].split(",")]
        except:
            friends=[int(0)]
        tweets = [int(i) for i in row[2].split(",")]
        user_friends[row[0]] = friends
        user_tweets [row[0]] = tweets
        cnt = max(cnt, max(friends))



    return user_friends,user_tweets, cnt + 1





class Word(object):
    def load_tweets(self, df, max_len=None):
        text = list(df['tweet'].values)
        tid = list(df['tweetId'].values)
        print("Build Vocabulary!")

        text = [str(t).split(" ") for t in tqdm(text)]

        idx2word = sorted(set([y for x in text for y in x]))  # flatten
        word2idx = {idx2word[idx]: idx for idx in range(len(idx2word))}
        pad = len(idx2word)  #padding is not zero.

        # vectorize
        max_len = max([len(sen) for sen in text]) if max_len is None else max_len
        tweet = np.full(shape=(len(tid), max_len), fill_value=pad, dtype=np.int64)
        for row in range(len(text)):
            for col in range(len(text[row])):
                if col < max_len:
                    tweet[tid[row], col] = word2idx[text[row][col]]

        self.idx2word = idx2word
        self.word2idx = word2idx
        self.pad = pad
        self.max_len = max_len

        print(len(idx2word))

        return tweet

    def vectorize_from_clean(self, df, max_len=None):

        text = list(df.values)
        print("Build Vocabulary!")

        text = [eval(t) for t in tqdm(text)]

        idx2word = sorted(set([y for x in text for y in x]))  # flatten
        word2idx = {idx2word[idx]: idx for idx in range(len(idx2word))}
        pad = len(idx2word)

        # vectorize
        max_len = max([len(sen) for sen in text]) if max_len is None else max_len
        tweet = np.full(shape=(len(text), max_len), fill_value=pad, dtype=np.int64)
        for row in range(len(text)):
            for col in range(len(text[row])):
                if col < max_len:
                    tweet[row, col] = word2idx[text[row][col]]
        tweet = [tuple(t) for t in list(tweet)]

        self.idx2word = idx2word
        self.word2idx = word2idx
        self.pad = pad
        self.max_len = max_len

        return tweet

    def vectorize_from_raw(self, df, max_len=None):
        """

        :param df: DataFrame of Tweet, which is raw language text
        :return: DataFrame of Word Index Vector
        """
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from time import time


        wordnet_lemmatizer = WordNetLemmatizer()

        stopworddic = set(stopwords.words('english'))

        text = list(df.values)

        # tokenize
        print("Sing!")
        print("Tokenization!")
        text = [nltk.word_tokenize(t) for t in tqdm(text)]

        # clean the stopword
        # print("Clean Stop Words!")
        # text = [ [i for i in t if i not in stopworddic] for t in tqdm(text)]

        # lemmatizer
        print("Lemmatization!")
        text = [[wordnet_lemmatizer.lemmatize(i) for i in t] for t in tqdm(text)]

        # # build vocabulary
        # print("Build Vocabulary!")
        # idx2word = sorted(set([y for x in text for y in x])) # flatten
        # word2idx = {idx2word[idx]: idx for idx in range(len(idx2word))}
        # pad = len(idx2word)

        # vectorize
        # max_len = max([len(sen) for sen in text]) if max_len is None else max_len
        # tweet = np.full(shape=(len(text), max_len), fill_value=pad, dtype=np.int64)
        # for row in range(len(text)):
        #     for col in range(len(text[row])):
        #         if col < max_len:
        #             tweet[row, col] = word2idx[text[row][col]]
        # tweet = list(tweet)
        # tweet = [tuple(t) for t in list(tweet)]
        #
        # self.idx2word = idx2word
        # self.word2idx = word2idx
        # self.pad = pad
        # self.max_len = max_len

        return text




