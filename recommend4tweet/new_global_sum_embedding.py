import torch
import torch.nn.functional as F
import torch.nn as nn
from engine import Engine
from utils import use_cuda, resume_checkpoint
import numpy as np
from torch.autograd import Variable

'''
LSTM + Supervised attention with TF-IDF

'''

class New_Gloabl_sum_embedding(torch.nn.Module):
    def __init__(self, config):
        super(New_Gloabl_sum_embedding, self).__init__()
        self.config = config
        self.use_gpu = torch.cuda.is_available()

        self.args = config['args']
        self.vocab = config['vocab']
        self.n_input = self.args.embedding_dim  # 100
        self.n_steps = config['args'].max_seq_len
        self.batch_size = self.args.batch_size

        # GROUP
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim_mf']
        self.num_friends = config['num_friends']
        self.user_friends = config['user_friends']

        #GMF
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.user_friend_indices, self.friends_real = self.init_user_friend_indices()  # [USERID, FRIEND-ID-LIST]
        self.tweet_word_indices = self.init_tweet_word()



        # LSTM
        self.hidden_dim = self.args.hidden_dim  # 128
        self.weight_item_lstm = torch.nn.Linear(in_features=self.latent_dim,
                                        out_features=self.hidden_dim)
        self.hidden2mf_lstm = nn.Linear(self.hidden_dim, self.latent_dim)
        self.lstm = nn.LSTM(self.args.embedding_dim, self.hidden_dim)
        self.hidden_lstm = self.init_hidden()
        self.mean = self.args.__dict__.get("lstm_mean", True)
        self.word_embeddings_lstm = nn.Embedding(self.vocab.pad + 1, self.n_input, padding_idx=self.vocab.pad)

        # GROUP
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_friends + 1, embedding_dim=self.latent_dim,
                                                 padding_idx=self.num_friends)
        self.tanh = torch.nn.Tanh()

        # Weighted sum output
        self.three_weight1 = torch.nn.Linear(in_features=self.latent_dim, out_features=int(self.latent_dim / 2))
        self.three_weight2 = torch.nn.Linear(in_features=int(self.latent_dim / 2),
                                             out_features=int(self.latent_dim / 4))
        self.three_weight3 = torch.nn.Linear(in_features=int(self.latent_dim / 4), out_features=2)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)

        self.logistic = torch.nn.Sigmoid()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, user_indices, tweet_indices, item_indices, group=True):

        # lstm---tweets
        item_embedding_tweet = self.weight_item_lstm(self.embedding_item(item_indices))
        sentence = self.tweet_word_indices[tweet_indices]
        tweet_embeds = self.word_embeddings_lstm(sentence)  # [b,s,e]64x200x300
        x = tweet_embeds.permute(1, 0, 2)  # 200x64x300 [s,b,e]
        self.hidden_lstm = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden_lstm = self.lstm(x, self.hidden_lstm)  # 200x64x128 [s,b,e]
        if self.mean == "mean":
            out = lstm_out.permute(1, 0, 2)  # [b,s,e]
            h = torch.mean(out, 1)  # [b,e]
        else:
            h = lstm_out[-1]  # [b,e]


        element_product_tweet = self.hidden2mf_lstm(h * item_embedding_tweet)

        # GROUP
        item_embedding = self.embedding_item(item_indices)
        friend_indices = self.user_friend_indices[user_indices]
        friend_embedding = self.embedding_user(friend_indices)  # (b, f, e)
        friend_num = self.config['friend_clip'] - (friend_indices == self.num_friends).sum(dim=1, keepdim=True).float()

        if group:
            group_idx = self.logistic(
                self.affine_output(torch.unsqueeze(item_embedding, 1) * friend_embedding))  # [b,f,1]
            friend_embedding = friend_embedding * group_idx  # [b,f,e]
        user_embedding = friend_embedding.sum(dim=1) / friend_num
        element_product_grouping = user_embedding  # [b,e]

        # Joint
        self.alpha = self.logistic(self.three_weight1(item_embedding))
        self.alpha = self.logistic(self.three_weight2(self.alpha))
        self.alpha = self.logistic(self.three_weight3(self.alpha))

        self.mean_alpha = torch.mean(self.alpha, 0)  # [3]
        self.alpha = torch.split(self.alpha, 1, 1)
        vector = self.alpha[0] * element_product_tweet + self.alpha[1] * element_product_grouping # [b,e]

        logits = self.affine_output(vector * item_embedding)
        rating = torch.squeeze(self.logistic(logits))

        return (rating, group_idx)

    def init_weight(self):
        self.attention_item.weight.data = torch.zeros(self.num_items, self.latent_dim)

    def init_tweet_word(self):
        tweet_word_indices = torch.LongTensor(self.args.tweet, device=self.args.device_id)
        if self.args.use_cuda:
            tweet_word_indices = tweet_word_indices.cuda()
        return tweet_word_indices

    def init_user_friend_indices(self):

        clip = self.config['friend_clip']
        friends_real = np.zeros((self.num_users, clip), dtype=int)
        user_friend_indices = np.full(shape=(self.num_users, clip), fill_value=self.num_friends)
        for u in range(self.num_users):
            friends = self.user_friends[u][:clip]
            user_friend_indices[u][range(len(friends))] = friends
            friends_real[u][range(len(friends))] = 1

        friends_real = torch.FloatTensor(friends_real, device=self.config['device_id'])
        user_friend_indices = torch.LongTensor(user_friend_indices, device=self.config['device_id'])
        if self.config['use_cuda']:
            user_friend_indices = user_friend_indices.cuda()
            friends_real = friends_real.cuda()

        return user_friend_indices, friends_real


class New_Gloabl_sum_embeddingEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        '''

        self.model = New_Gloabl_sum_embedding(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(New_Gloabl_sum_embeddingEngine, self).__init__(config)
        if config['pretrain']:
            print("load pretrained model...")
            self.model.load_pretrain_weights()

        if config['pretrain_grouping']:
            print("load pretrained grouping embedding")
            self.model.load_pretrain_grouping()
        print(self.model)
             '''

        self.model = torch.load(config['pretrain_dir'])