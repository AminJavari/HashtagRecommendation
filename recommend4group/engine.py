import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK

import scipy.sparse
import numpy as np

from time import time


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.sparse = False
        if config['friend_item_matrix'].split(".")[-1] == "npz":
            self.friend_item_matrix = scipy.sparse.load_npz(config['friend_item_matrix'])
            self.sparse = True
        else:
            self.friend_item_matrix = np.load(config['friend_item_matrix'])

    def shape_friend_ground_truth(self, users, items):
        friend_list = self.model.user_friend_indices[users].cpu().numpy()
        item_list = items.cpu().numpy()
        if self.sparse:
            friend_gt = []
            for friends, item in zip(friend_list, item_list):
                friend_gt.append(
                    (np.array(self.friend_item_matrix[item, friends].todense()) / self.friend_item_matrix[item, 0])[0])
        else:
            friend_gt = self.friend_item_matrix[item_list[:, None], friend_list] / (
                        self.friend_item_matrix[item_list[:, None], 0] + 1e-12)
        friend_gt = torch.FloatTensor(friend_gt)
        if self.config['use_cuda'] is True:
            friend_gt = friend_gt.cuda()
        return friend_gt

    def train_single_batch(self, users, items, ratings, friend_gt):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred, group_idx = self.model(users, items)
        # loss = self.crit(ratings_pred, ratings) + 0.01*self.mse(torch.squeeze(group_idx), friend_gt)
        l2_reg = None
        for n, p in self.model.named_parameters():
            if "attention" not in n:
                if l2_reg is None:
                    l2_reg = p.norm(2)
                else:
                    l2_reg = l2_reg + p.norm(2)

        loss = self.crit(ratings_pred, ratings) + l2_reg * self.config['l2_other'] + self.config['alpha']*self.crit(torch.squeeze(group_idx).view(-1), friend_gt.view(-1))
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        t1 = time()
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            rating = rating.float()

            if user.size(0) != self.model.config['batch_size']:
                break

            if self.config['use_cuda'] is True:
                user = user.cuda()
                item = item.cuda()
                rating = rating.cuda()
            loss = self.train_single_batch(user, item, rating, self.shape_friend_ground_truth(user, item))
            if batch_id % 1000 == 0:
                t2 = time()
                print('[Training Epoch {}] Batch {}, Loss {:.4f}, Time {:.2f} '.format(epoch_id, batch_id, loss, t2-t1))
                t1 = time()
                print(torch.norm(self.model.attention_item.weight))
            total_loss += loss

        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def test_epoch(self, test_users, test_items):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            batch_size = self.model.config['batch_size']
            batch_num = (len(test_users) // batch_size) + 1
            score = []
            for i in range(batch_num):
                test_user_batch, test_item_batch = test_users[batch_size*i: batch_size*(i+1)], test_items[batch_size*i: batch_size*(i+1)]
                score_batch = self.model(test_user_batch, test_item_batch)[0]
                score_batch = score_batch if len(score_batch.size()) else score_batch.unsqueeze(-1)
                score.append(score_batch)
        return torch.cat(score, -1)

    def evaluate(self, evaluate_data, epoch_id, save=True):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        test_users, test_items = Variable(evaluate_data[0]), Variable(evaluate_data[1])
        negative_users, negative_items = Variable(evaluate_data[2]), Variable(evaluate_data[3])
        test_keys, negative_keys = evaluate_data[4], evaluate_data[5]
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
        test_scores = self.test_epoch(test_users, test_items)
        negative_scores = self.test_epoch(negative_users, negative_items)
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist(),
                                 test_keys, negative_keys]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        if save:
            self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
            self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg


    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
