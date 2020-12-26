import torch
import torch.nn.functional as F
import torch.nn as nn
from engine import Engine
from utils import use_cuda, resume_checkpoint
import numpy as np
from torch.autograd import Variable


'''
memory+dynamic grouping with shared item embedding



'''



class New_Gloabl_sum_embedding(torch.nn.Module):
    def __init__(self, config):
        super(New_Gloabl_sum_embedding, self).__init__()
        self.config = config
        self.user_tweets = config['user_tweets']
        self.args = config['args']
        self.vocab = config['vocab']
        self.n_input = self.args.embedding_dim
        self.n_sequence = config['args'].max_tweets_num
        self.n_steps = config['args'].max_seq_len
        self.hoops = self.args.hoops
        # GROUP
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim_mf']
        self.num_friends = config['num_friends']
        self.user_friends = config['user_friends']


        # memory
        self.embedding_word= torch.nn.Embedding(num_embeddings=self.config['vocab'].pad + 1,
                                                   embedding_dim=self.n_input, padding_idx=self.config['vocab'].pad)

        self.embedding_item= torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.weight_memory = torch.nn.Linear(in_features=self.latent_dim, out_features=self.n_input)
        self.weight_fc_word = torch.nn.Linear(in_features=self.n_input, out_features=1)
        self.weight_fc_sentence = torch.nn.Linear(in_features=self.n_input, out_features=1)
        self.transfer_to_latent_dim = torch.nn.Linear(in_features=self.n_input, out_features=self.latent_dim)


        # GROUP

        self.user_tweet_indices = self.init_user_tweet_indices()
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_friends+1, embedding_dim=self.latent_dim, padding_idx=self.num_friends)
        self.tanh = torch.nn.Tanh()
        self.user_friend_indices, self.friends_real = self.init_user_friend_indices()  # [USERID, FRIEND-ID-LIST]

        # Weighted sum output
        self.three_weight1 = torch.nn.Linear(in_features=self.latent_dim, out_features=int(self.latent_dim / 2))
        self.three_weight2 = torch.nn.Linear(in_features=int(self.latent_dim / 2),
                                             out_features=int(self.latent_dim / 4))
        self.three_weight3 = torch.nn.Linear(in_features=int(self.latent_dim / 4), out_features=2)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()




    def forward(self, user_indices, item_indices, group=True):

        # memory---tweets
        item_embedding_initial = self.weight_memory(self.embedding_item(item_indices))  # [b,e]

        for i in range(self.hoops):
            if i == 0:
                item_embedding_memory = item_embedding_initial
            else:
                item_embedding_memory = output

            # word level attention
            item_embedding_B = torch.unsqueeze(item_embedding_memory, 1)  # [b,1,e]
            item_embedding_B = torch.unsqueeze(item_embedding_B, 1)  # [b,1,1,e]
            item_embedding_B = item_embedding_B.repeat(1, self.n_sequence, self.n_steps, 1)  # [b,t,w,e]

            tweets_vector = self.embedding_word(self.user_tweet_indices[user_indices])  # [b,t,w,e]

            word_weight = torch.sum(torch.mul(tweets_vector, item_embedding_B), 3)  # [b,t,w]
            word_weight = F.softmax(word_weight, dim=2)  # [b,t,w]
            word_weight = torch.unsqueeze(word_weight, 3)  # [b,t,w,1]
            word_weight = word_weight.repeat(1, 1, 1, self.n_input)  # [b,t,w,e]

            sentence = torch.mul(word_weight, tweets_vector)  # [b,t,w,e]
            sentence = torch.sum(sentence, 2)  # [b,t,e]

            # sentence level attention
            sentence_weight = self.weight_fc_word(sentence)  # [b,t,1]
            item_embedding_sentence = torch.unsqueeze(item_embedding_memory, 1)  # [b,1,e]
            item_embedding_sentence = item_embedding_sentence.repeat(1, self.n_sequence, 1)  # [b,t,e]
            item_weight = self.weight_fc_sentence(item_embedding_sentence)  # [b,t,1]
            sentence_weight_final = self.tanh(sentence_weight + item_weight)  # [b,t,1]
            sentence_weight_final = torch.reshape(sentence_weight_final, (-1, self.n_sequence))  # [b,t]
            sentence_weight_final = F.softmax(sentence_weight_final, dim=1)  # [b,t]
            sentence_weight_final = torch.unsqueeze(sentence_weight_final, 2)  # [b,t,1]
            sentence_weight_final = sentence_weight_final.repeat(1, 1, self.n_input)  # [b,t,e]
            user_representation = torch.sum(torch.mul(sentence, sentence_weight_final), 1)  # [b,e]

            output = user_representation + item_embedding_memory  # [b,e]

        element_product_tweets = self.transfer_to_latent_dim(output) # [b,e]



        # GROUP

        friend_indices = self.user_friend_indices[user_indices]
        friend_embedding =  self.embedding_user(friend_indices) # (b, f, e)
        friend_num = self.config['friend_clip'] - (friend_indices == self.num_friends).sum(dim=1, keepdim=True).float()

        if group:
            group_idx = self.logistic(self.affine_output(torch.unsqueeze(self.embedding_item(item_indices), 1) * friend_embedding)) #[b,f,1]
            friend_embedding = friend_embedding * group_idx #[b,f,e]
        user_embedding = friend_embedding.sum(dim=1) / friend_num
        item_embedding = self.embedding_item(item_indices)
        element_product_grouping = user_embedding #[b,e]

        # Joint

        self.alpha = self.logistic(self.three_weight1(item_embedding))
        self.alpha = self.logistic(self.three_weight2(self.alpha))
        self.alpha = self.logistic(self.three_weight3(self.alpha))
        self.mean_alpha = torch.mean(self.alpha, 0)  # [3]
        self.alpha = torch.split(self.alpha, 1, 1)
        vector = self.alpha[0] * element_product_tweets + self.alpha[1] * element_product_grouping  # [b,e]

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

    def init_user_tweet_indices(self):
        tweet_refer = self.config['args'].tweet
        user_tweet_indices = np.full(shape=(self.num_users, self.n_sequence, self.config['args'].max_seq_len),
                                     fill_value=self.config['vocab'].pad)  # [u,t
        for u in range(self.num_users):
            tweets_list_ids = self.user_tweets[u][:self.config['args'].max_tweets_num]
            tweets_user = tweet_refer[tweets_list_ids]  # [t,w]
            user_tweet_indices[u][range(tweets_user.shape[0])] = tweets_user  # [u,t,w]
        user_tweet_indices = torch.LongTensor(user_tweet_indices, device=self.config['device_id'])
        if self.config['use_cuda']:
            user_tweet_indices = user_tweet_indices.cuda()
        print("tweets_information loaded!")
        return user_tweet_indices




class New_Gloabl_sum_embedding_shareEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = New_Gloabl_sum_embedding(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(New_Gloabl_sum_embedding_shareEngine, self).__init__(config)
        if config['pretrain']:
            print("load pretrained model...")
            self.model.load_pretrain_weights()

        if config['pretrain_grouping']:
            print("load pretrained grouping embedding")
            self.model.load_pretrain_grouping()
        print(self.model)