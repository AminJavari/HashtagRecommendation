import torch
import torch.nn.functional as F
from engine import Engine
from utils import use_cuda, resume_checkpoint
import numpy as np


class Grouping(torch.nn.Module):
    def __init__(self, config):
        super(Grouping, self).__init__()
        self.num_users = config['num_friends_pretrain']
        self.num_items = config['num_items_pretrain']
        self.latent_dim = config['latent_dim']
        self.config = config

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc = torch.nn.Linear(in_features=self.latent_dim, out_features=1)

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        score = self.fc(element_product)
        return torch.squeeze(score)

class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.num_friends = config['num_friends']
        self.user_friends = config['user_friends']
        self.config = config
        # self.user_friend_hots = self.init_user_friend()

        # grouping module
        self.attention_user = torch.nn.Embedding(num_embeddings=self.num_friends+1, embedding_dim=self.latent_dim,padding_idx=self.num_friends)
        self.attention_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.attention_fc = torch.nn.Linear(in_features=self.latent_dim, out_features=1)

        for p in self.parameters():
            p.requires_grad = True

        self.user_friend_indices = self.init_user_friend_indices()

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_friends+1, embedding_dim=self.latent_dim, padding_idx=self.num_friends)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.user_bias = torch.nn.Parameter(torch.zeros(self.latent_dim))

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        # Pretrain
        if config['deepwalk_embedding'] is not None:
            print("load deepwalk embedding!")
            pretrain = torch.FloatTensor(np.load(config['deepwalk_embedding']))
            self.embedding_user.weight.data[:pretrain.size(0), :] = torch.FloatTensor(np.load(config['deepwalk_embedding']))

        # self.init_weight()

    def forward(self, user_indices, item_indices, group=True):
        friend_indices = self.user_friend_indices[user_indices]
        friend_embedding =  self.embedding_user(friend_indices) # (b, f, e)
        friend_num = self.config['friend_clip'] - (friend_indices == self.num_friends).sum(dim=1, keepdim=True).float()
        group_idx = self.logistic(self.affine_output(torch.unsqueeze(self.embedding_item(item_indices), 1) * friend_embedding))
        if group:
            friend_embedding = friend_embedding * group_idx
        # user_embedding = friend_embedding.sum(dim=1) / group_idx.sum(dim=1)+1e-12   # (b, 1, e) * (b, f, e)
        user_embedding = friend_embedding.sum(dim=1) / friend_num
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = torch.squeeze(self.logistic(logits))

        return (rating, group_idx)

    def init_weight(self):
        self.attention_item.weight.data = torch.zeros(self.num_items, self.latent_dim)


    def init_user_friend_indices(self):
        clip = self.config['friend_clip']
        user_friend_indices = np.full(shape=(self.num_users, clip), fill_value=self.num_friends)
        for u in range(self.num_users):
            friends = self.user_friends[u][:clip]
            user_friend_indices[u][range(len(friends))] = friends
        user_friend_indices = torch.LongTensor(user_friend_indices, device=self.config['device_id'])
        if self.config['use_cuda']:
            user_friend_indices = user_friend_indices.cuda()
        return user_friend_indices

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        config = self.config
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
        self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item.weight.data = gmf_model.embedding_item.weight.data
        self.affine_output.weight.data = gmf_model.affine_output.weight.data
        self.affine_output.bias.data = gmf_model.affine_output.bias.data


    def load_pretrain_grouping(self):
        """Loading weights from trained GMF model"""
        config = self.config
        grouping_module = Grouping(config)
        if config['use_cuda'] is True:
            grouping_module.cuda()
        resume_checkpoint(grouping_module, model_dir=config['pretrain_grouping_module'], device_id=config['device_id'])

        self.attention_user.weight.data[:grouping_module.num_users] = grouping_module.embedding_user.weight.data[:grouping_module.num_users]
        self.attention_item.weight.data[:grouping_module.num_items] = grouping_module.embedding_item.weight.data[:grouping_module.num_items]

        self.attention_fc.weight = grouping_module.fc.weight
        self.attention_fc.bias = grouping_module.fc.bias


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)
        if config['pretrain']:
            print("load pretrained model...")
            self.model.load_pretrain_weights()

        if config['pretrain_grouping']:
            print("load pretrained grouping embedding")
            self.model.load_pretrain_grouping()
        print(self.model)
