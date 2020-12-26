import pandas as pd
import numpy as np
from gmf import GMFEngine
from data import SampleGenerator
from utils import load_friends
import sys
import opt2 as opts
import numpy as np
import torch

args = opts.parse_opt()
print(args)

config = {'alias': args.alias,
          'num_epoch': args.max_epoch,
          'batch_size': args.batch_size,
          'optimizer': 'adam',
          'adam_lr': args.learning_rate,
          'latent_dim_mf': args.latent_dim_mf,
          'latent_dim_mlp': args.latent_dim_mlp,
          'friend_clip': 400,
          'num_negative': args.num_negative,
          'l2_regularization': args.l2,
          'use_cuda': True,
          'device_id': 0,
          'friend_item_matrix': args.group_weight,
          'layers': eval(args.layers),
          'pretrain': args.pretrain,
          'pretrain_mf': '../recommend4group/checkpoints/{}'.format('user_no_pretrain_Epoch40_HR0.4471_NDCG0.2670.model'),
          'pretrain_mlp': 'checkpoints/{}'.format('mlp-params.model'),
          'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
          'pretrain_grouping_module':'../friend2tag/checkpoints/{}'.format('test_Epoch0_HR172.8309_NDCG172.8309.model'),
          # 'pretrain_grouping_module':'../friend2tag/checkpoints/{}'.format('user_Epoch39_HR0.5263_NDCG0.5263.model'),
          'pretrain_grouping': args.pretrain_grouping,
          'lr_ratio': args.learning_rate_ratio,
          'l2_other': args.l2_other,
          'alpha': args.alpha,
          # 'deepwalk_embedding': "../recommend4user/embedding.npy",
          'deepwalk_embedding': None,
          }


# Load Rating Data
# Load Rating Data
if args.data_train.find("for_user") != -1:
    data_rating_train = pd.read_csv(args.data_train, sep=',', header=0, names=['userId', 'itemId'],  engine='python', dtype= np.int)
    data_rating_test = pd.read_csv(args.data_test, sep=',', header=0, names=['userId', 'itemId', 'negative_samples'],  engine='python', dtype= {'userId':np.int, 'itemId':np.int})
else:
    data_rating_train = pd.read_csv(args.data_train, sep=',', header=0, names=['userId', 'itemId', 'tweetId'],  engine='python', dtype= np.int)
    data_rating_test = pd.read_csv(args.data_test, sep=',', header=0, names=['userId', 'itemId', 'tweetId', 'negative_samples'],  engine='python', dtype= {'userId': np.int, 'itemId': np.int, 'tweetId': np.int})

data_rating_test['negative_samples'] = data_rating_test['negative_samples'].apply(lambda x: eval("["+x+"]"))

data_rating_train['rating'] = np.ones(len(data_rating_train))
data_rating_test['rating'] = np.ones(len(data_rating_test))

data_rating = data_rating_train.append(data_rating_test)

print('Range of userId is [{}, {}]'.format(data_rating.userId.min(), data_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(data_rating.itemId.min(), data_rating.itemId.max()))

# Read the grouping information
if args.pretrain_grouping:
    data_grouping = pd.read_csv(args.data_grouping, sep=",", header=0, names=['friendId', 'tagId', 'score'],engine='python')
    config['num_friends_pretrain'] = int(data_grouping.friendId.max() + 1)
    config['num_items_pretrain'] = int(data_grouping.tagId.max() + 1)
    del data_grouping

# DataLoader for training
sample_generator = SampleGenerator(ratings=data_rating, train=data_rating_train, test=data_rating_test)

config['num_users'], config['num_items'] = int(data_rating.userId.max() + 1), int(data_rating.itemId.max() + 1)
config['user_friends'], config['num_friends'] = load_friends(args.data_friends)

# Specify the exact model
model = sys.argv[1] if len(sys.argv) == 2 else "gmf"
if args.model.lower() == "gmf":
    config['latent_dim'] = config['latent_dim_mf']
    engine = GMFEngine(config)
elif args.model.lower() == "mlp":
    config['latent_dim'] = config['latent_dim_mlp']
    engine = MLPEngine(config)
elif args.model.lower() == "neumf":
    engine = NeuMFEngine(config)

# Train this model
evaluate_data = sample_generator.evaluate_data
sample_train_data = sample_generator.sample_train_data

print("TRAINING:", end=" ")
engine.evaluate(sample_train_data, epoch_id=0, save=False)
print("TESTING :", end=" ")
hit_ratio_max, ndcg_max = engine.evaluate(evaluate_data, epoch_id=0)

torch.save(engine.model, args.alias + ".p")

for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    print("TRAINING:", end=" ")
    engine.evaluate(sample_train_data, epoch, save=False)
    print("TESTING :", end=" ")
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    if hit_ratio_max <= hit_ratio or ndcg_max <= ndcg:
        hit_ratio_max = max(hit_ratio_max, hit_ratio)
        ndcg_max = max(ndcg_max, ndcg)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
        torch.save(engine.model, args.alias + ".p")

