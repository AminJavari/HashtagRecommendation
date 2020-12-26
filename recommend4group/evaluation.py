import torch
import numpy as np

from tqdm import tqdm


class Grouping(torch.nn.Module):
    def __init__(self, config):
        super(Grouping, self).__init__()
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.num_friends = config['num_friends']
        self.config = config

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_friends+1, embedding_dim=self.latent_dim, padding_idx=self.num_friends)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        score = self.logistic(self.affine_output(element_product))
        return torch.squeeze(score)

# load the pretrained model
model_dir = "../embedding/gmf_mean_user_Epoch15_HR0.4147_NDCG0.2433.model"
save_dir = "gmf_4user_friends_tag.npy"
device_id = 0

pretrained_dict = torch.load(model_dir, map_location=lambda storage, loc: storage.cuda(device=device_id))

# load the grouping model
config = {}
config['num_items'] = pretrained_dict['embedding_item.weight'].size()[0]
config['latent_dim'] = pretrained_dict['embedding_user.weight'].size()[1]
config['num_friends'] = pretrained_dict['embedding_user.weight'].size()[0] - 1

model = Grouping(config)
model.cuda()
model_dict = model.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# predict
mat = np.zeros(shape=(config['num_friends'], config['num_items']))
model.eval()
with torch.no_grad():
    for i in tqdm(range(config['num_friends'])):
        user_indices = torch.LongTensor([i for _ in range(config['num_items'])]).cuda()
        item_indices = torch.LongTensor([j for j in range(config['num_items'])]).cuda()
        output = model(user_indices, item_indices)
        mat[i] = output.cpu().numpy()

# save the rating matrix
np.save(save_dir, mat)