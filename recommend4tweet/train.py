import pandas as pd
import numpy as np
from new_global_sum_embedding_gmf import  New_Gloabl_sum_embedding_gmfEngine
from lstm_mlp import LSTMMLPEngine
from gmf import GMFEngine
from data import SampleGenerator
from utils import load_friends, Word
from new_global_sum_embedding import New_Gloabl_sum_embeddingEngine
from new_global_sum_embedding_mlp import New_Gloabl_sum_embeddingMLPEngine
import opts

args = opts.parse_opt()
print(args)

config = {'alias': args.alias,
        'num_epoch': args.max_epoch,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'optimizer': 'adam',
        'adam_lr': args.learning_rate,
        'latent_dim_mf': args.latent_dim_mf,
        'friend_clip': 400,
        'latent_dim_mlp': args.latent_dim_mlp,
        'num_negative': 4,
        'friend_item_matrix':'../data/Twitter/friend_tag_for_'+args.type+'_tfidf.npy',
        'layers': eval(args.layers),  # layers[0] is the concat of latent user vector & latent item vector
        'l2_attention': args.l2_attention,
        'l2_regularization': 0,
        'use_cuda': True,
        'device_id': args.gpu,
        'pretrain': args.pretrain,
        'pretrain_mf': 'checkpoints/{}'.format('gmf-params.model'),
        'pretrain_mlp': 'checkpoints/{}'.format('mlp-params.model'),
        'model_dir':'checkpoints_Feb/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
        'pretrain_grouping_module': '../friend2tag/checkpoints/{}'.format('test_Epoch0_HR172.8309_NDCG172.8309.model'),
        'pretrain_grouping': args.pretrain_grouping,
          'pretrain_dir': 'checkpoints/twitter_lr5e-3_l2a1e-5_l2l0_Epoch11_HR0.7663_NDCG0.5793.model',
        'lr_ratio': args.learning_rate_ratio,
        'l2_lstm': args.l2_lstm}


# Load Rating Data
data_rating_train = pd.read_csv(args.data_train, sep=',', header=0, names=['userId', 'itemId', 'tweetId'],  engine='python', dtype= np.int)
data_rating_test = pd.read_csv(args.data_test, sep=',', header=0, names=['userId', 'itemId', 'tweetId', 'negative_samples'],  engine='python', dtype= {'userId': np.int, 'itemId': np.int, 'tweetId': np.int})
data_rating_test['negative_samples'] = data_rating_test['negative_samples'].apply(lambda x: eval("["+x+"]"))
data_tweet = pd.read_csv(args.data_tweets, sep=',', header=0, names=['tweetId', 'tweet'],  engine='python', dtype={'tweetId':  np.int, 'tweet': str})

data_rating_train['rating'] = np.ones(len(data_rating_train))
data_rating_test['rating'] = np.ones(len(data_rating_test))

data_rating = data_rating_train.append(data_rating_test)

print('Range of userId is [{}, {}]'.format(data_rating.userId.min(), data_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(data_rating.itemId.min(), data_rating.itemId.max()))
print('Range of tweetId is [{}, {}]'.format(data_rating.tweetId.min(), data_rating.tweetId.max()))

# Read the grouping information
if args.pretrain_grouping:
    data_grouping = pd.read_csv(args.data_grouping, sep=",", header=0, names=['friendId', 'tagId', 'score'],engine='python')
    config['num_friends_pretrain'] = int(data_grouping.friendId.max() + 1)
    config['num_items_pretrain'] = int(data_grouping.tagId.max() + 1)
    del data_grouping

args.item_num = int(data_rating.itemId.max() + 1)

# Process the tweet
vocab = Word()
tweet = vocab.load_tweets(data_tweet, max_len=200)

# Read the grouping information
data_grouping = pd.read_csv(args.data_grouping, sep=",", header=0, names=['friendId', 'tagId', 'score'],engine='python')
config['num_friends_pretrain'] = int(data_grouping.friendId.max() + 1)
config['num_items_pretrain'] = int(data_grouping.tagId.max() + 1)
del data_grouping

# config
config['num_users'], config['num_items'] = int(data_rating.userId.max() + 1), int(data_rating.itemId.max() + 1)
config['user_friends'], config['num_friends'] = load_friends(args.data_friends)
args.tweet = tweet
config['args'] = args
config['vocab'] = vocab

# Specify the exact model

if args.model.lower() == "lstm-mlp":
    config['model_type'] = 1
    engine = LSTMMLPEngine(config)
elif args.model.lower() == "gmf":
    config['model_type'] = 0
    engine = GMFEngine(config)
elif args.model.lower() == "global_sum_embedding":
    config['model_type'] = 3
    config['latent_dim'] = config['latent_dim_mf']
    walk_length = args.walk_length
    engine = New_Gloabl_sum_embeddingEngine(config)
elif args.model.lower() == "global_sum_embedding_gmf":
    config['model_type'] = 3
    config['latent_dim'] = config['latent_dim_mf']
    walk_length = args.walk_length
    engine = New_Gloabl_sum_embedding_gmfEngine(config)
elif args.model.lower() == "global_sum_embedding_mlp":
    config['model_type'] = 3
    config['latent_dim'] = config['latent_dim_mf']
    walk_length = args.walk_length
    engine = New_Gloabl_sum_embeddingMLPEngine(config)




# DataLoader for training
sample_generator = SampleGenerator(ratings=data_rating, train=data_rating_train, test=data_rating_test)

# Train this model
evaluate_data = sample_generator.evaluate_data
sample_train_data = sample_generator.sample_train_data

print("TRAINING:")
engine.evaluate(sample_train_data, epoch_id=0, save=False)
print("TESTING :")
hit_ratio_max, ndcg_max = engine.evaluate(evaluate_data, epoch_id=0)


for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    #print("TRAINING:")
    #engine.evaluate(sample_train_data, epoch, save=False)
    print("TESTING :")
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    if hit_ratio_max <= hit_ratio or ndcg_max <= ndcg:
        hit_ratio_max = max(hit_ratio_max, hit_ratio)
        ndcg_max = max(ndcg_max, ndcg)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
