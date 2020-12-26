import pandas as pd
import numpy as np
from gmf import GMFEngine
from gmf_lstm import GMFLSTMEngine
from New_global_sum_embedding import New_Gloabl_sum_embedding_shareEngine
from new_global_sum_embedding_gmf import New_Gloabl_sum_embedding_gmfEngine
from New_global_sum_embedding_MLP import New_Gloabl_sum_embedding_MLPEngine
from data import SampleGenerator
from utils import load_friends_tweets,Word
import sys
import opt2


args = opt2.parse_opt()
print(args)

config = {'alias': args.alias,
          'num_epoch': args.max_epoch,
          'batch_size': args.batch_size,
          'optimizer': 'adam',
          'alpha': args.alpha,
          'adam_lr': args.learning_rate,
          'latent_dim_mf': args.latent_dim_mf,
          'latent_dim_mlp': args.latent_dim_mlp,
          'friend_clip': 400,
          'num_negative': args.num_negative,
          'l2_regularization': 0,
          'use_cuda': True,
          'device_id': 0,
          'friend_item_matrix':'../data/Weibo/friend_tag_for_user_tfidf.npy',
          'layers': eval(args.layers),
          'pretrain': args.pretrain,
          'pretrain_mf': '../recommend4user/checkpoints/{}'.format('gmf-lr-1e-2_Epoch3_HR0.7137_NDCG0.3613.model'),
          'pretrain_mlp': 'checkpoints_Feb/{}'.format('mlp-params.model'),
          'model_dir': 'checkpoints_Feb/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
          'group': args.group,
        'pretrain_grouping_module': '../friend2tag/checkpoints/{}'.format('test_Epoch0_HR172.8309_NDCG172.8309.model'),
          # 'pretrain_grouping_module':'../friend2tag/checkpoints/{}'.format('user_Epoch39_HR0.5263_NDCG0.5263.model'),
        'pretrain_grouping': args.pretrain_grouping,
        'lr_ratio': args.learning_rate_ratio,
        'l2_embedding': args.l2_embedding,
        'l2_attention': args.l2_attention,
          'lamda':args.lamda
          }

# Load Rating Data
data_rating_train = pd.read_csv(args.data_train, sep=',', header=0, names=['userId', 'itemId'],  engine='python', dtype= np.int)
data_rating_test = pd.read_csv(args.data_test, sep=',', header=0, names=['userId', 'itemId', 'negative_samples'],  engine='python', dtype= {'userId':np.int, 'itemId':np.int})
data_rating_test['negative_samples'] = data_rating_test['negative_samples'].apply(lambda x: eval("["+x+"]"))
data_tweet = pd.read_csv(args.data_tweets, sep=',', header=0, names=['tweetId', 'tweet'],  engine='python', dtype={'tweetId':  np.int,'tweet':str})

data_rating_train['rating'] = np.ones(len(data_rating_train)) #[userId, itemId]
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
    print ("group data reading finished!")

# Process the tweet
vocab = Word()
tweet = vocab.load_tweets(data_tweet, args.max_seq_len)
pad_word = vocab.pad
tweet_pad = np.full(shape=(1, args.max_seq_len), fill_value=pad_word, dtype=np.int64)
tweet = np.vstack([tweet, tweet_pad])

# config
config['num_users'], config['num_items'] = int(data_rating.userId.max() + 1), int(data_rating.itemId.max() + 1)
config['user_friends'],config['user_tweets'],config['num_friends'] = load_friends_tweets(args.data_profile)
args.tweet = tweet
config['args'] = args
config['vocab'] = vocab

# Specify the exact model
model = sys.argv[1] if len(sys.argv) == 2 else "gmf"
if args.model.lower() == "gmf":
    config['group'] = False
    config['latent_dim'] = config['latent_dim_mf']
    engine = GMFEngine(config)
elif args.model.lower() == "gmflstm":
    config['sum'] = False
    config['group'] = False
    config['latent_dim'] = config['latent_dim_mf']
    engine = GMFLSTMEngine(config)
elif args.model.lower() == "global_sum_embedding":

    config['latent_dim'] = config['latent_dim_mf']
    engine =New_Gloabl_sum_embedding_shareEngine(config)
elif args.model.lower() == "global_sum_embedding_gmf":

    config['latent_dim'] = config['latent_dim_mf']
    engine =New_Gloabl_sum_embedding_gmfEngine(config)


elif args.model.lower() == "global_sum_embedding_mlp":

    config['latent_dim'] = config['latent_dim_mf']
    engine =New_Gloabl_sum_embedding_MLPEngine(config)





# DataLoader for training
sample_generator = SampleGenerator(ratings=data_rating, train=data_rating_train, test=data_rating_test)

# Train this model
evaluate_data = sample_generator.evaluate_data
sample_train_data = sample_generator.sample_train_data

print("TRAINING:---------------------")
engine.evaluate(sample_train_data, epoch_id=0, save=False)
print("TESTING:----------------------")
hit_ratio_max, ndcg_max = engine.evaluate(evaluate_data, epoch_id=0)

for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    #print("TRAINING:-----------------")
    #engine.evaluate(sample_train_data, epoch, save=False)
    print("TESTING:------------------")
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    if hit_ratio_max <= hit_ratio or ndcg_max <= ndcg:
        hit_ratio_max = max(hit_ratio_max, hit_ratio)
        ndcg_max = max(ndcg_max, ndcg)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
