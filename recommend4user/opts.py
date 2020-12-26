import argparse, os
import configparser


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings

    parser.add_argument('--config', type=str, default="no_file_exists",
                        help='gpu number')
    parser.add_argument('--alias', type=str, default="try",
                        help='gpu number')
    parser.add_argument('--mlp_layers', nargs='?', default='[64,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")

    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden_dim')
    parser.add_argument('--hoops', type=int, default=2,
                        help='hoops for memory')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='max_seq_len')
    parser.add_argument('--max_tweets_num', type=int, default=50,
                        help='max_tweets_len')
    parser.add_argument('--max_friend_len', type=int, default=400,
                        help='max_seq_len')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='embedding_dim')
    parser.add_argument('--user_embedding_dim', type=int, default=100,
                        help='user_embedding_dim')
    parser.add_argument('--latent_dim_mf', type=int, default=32,
                        help='user_embedding_dim_mf')

    parser.add_argument('--latent_dim_mlp', type=int, default=32,
                        help='user_embedding_dim_mlp')
    parser.add_argument('--num_negative', type=int, default=4,
                        help='negtive sampling')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='learning_rate')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                        help='grad_clip')

    parser.add_argument('--model', type=str, default="global_sum_embedding_mlp",
                        help='model name')
    parser.add_argument('--friend', type=str, default="true",
                        help='need the following information')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='alpha * text + (1-alpha) * friend')

    parser.add_argument('--data_train', type=str, default="../data/Twitter/train_for_user.csv",
                        help='training set data dir')
    parser.add_argument('--data_test', type=str, default="../data/Twitter/test_for_user.csv",
                        help='testing set data dir')
    parser.add_argument('--data_profile', type=str, default="../data/Twitter/profile.csv",
                        help='profile data dir')
    parser.add_argument('--data_tweets', type=str, default="../data/Twitter/tweets.csv",
                        help='tweets data dir')
    parser.add_argument('--position', type=bool, default=False,
                        help='gpu number')

    parser.add_argument('--keep_dropout', type=float, default=0.8,
                        help='keep_dropout')
    parser.add_argument('--l2_embedding', type=float, default=1e-5,
                        help='l2 regularization')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='max_epoch')
    parser.add_argument('--embedding_file', type=str, default="glove.6b.300",
                        help='glove or w2v')
    parser.add_argument('--embedding_training', type=str, default="false",
                        help='embedding_training')
    parser.add_argument('--pretrain_embedding', type=str, default="false",
                        help='embedding_training')
    parser.add_argument('--pretrain', type=str, default="false",
                        help='embedding ncf')

    # kim CNN
    parser.add_argument('--kernel_sizes', type=str, default="1,2,3,5",
                        help='kernel_sizes')
    parser.add_argument('--kernel_nums', type=str, default="256,256,256,256",
                        help='kernel_nums')
    parser.add_argument('--embedding_type', type=str, default="non-static",
                        help='embedding_type')
    parser.add_argument('--lstm_mean', type=str, default="mean",  # last
                        help='lstm_mean')
    parser.add_argument('--lstm_layers', type=int, default=1,  # last
                        help='lstm_layers')
    parser.add_argument('--gpu', type=str, default='0,1,2,3',
                        help='gpu number')
    parser.add_argument('--proxy', type=str, default="null",
                        help='http://proxy.xx.com:8080')
    parser.add_argument('--debug', type=str, default="true",
                        help='gpu number')
    parser.add_argument('--layers', type=str, default='[16,64,32,16,8]',  # last
                        help='lstm-mlp layers')

    parser.add_argument('--embedding_dir', type=str, default=".glove/glove.6B.300d.txt",
                        help='embedding_dir')
    parser.add_argument('--from_torchtext', type=str, default="true",
                        help='from torchtext or native data loader')
    parser.add_argument('--log_dir', type=str, default="test",
                        help='log_dir')

    # grouping
    parser.add_argument('--pretrain_grouping_module', type=str, default=".glove/glove.6B.300d.txt",
                        help='embedding_dir')
    parser.add_argument('--data_grouping', type=str, default="../data/Twitter/friend_tag_for_user.csv",
                        help='get the information of grouping')
    parser.add_argument('--pretrain_grouping', type=str, default="false",
                        help='use the pretrain grouping weights or not')
    parser.add_argument('--learning_rate_ratio', type=float, default=0.1,
                        help='the lr ratio for grouping module compared with others')
    parser.add_argument('--l2_attention', type=float, default=1e-5,
                        help='the l2 reg on model weights except the grouping module')
    parser.add_argument('--group', type=str, default="True",
                        help='embedding group')



    #parameters of self-attention
    parser.add_argument('--freeze', type=bool, default=True, help='wehther to train the embeddings of friends.')
    parser.add_argument('--latent_dim_friends', type=int, default=64,
                        help='friends_embedding_dim')
    parser.add_argument('--alpha_relu', type=float, default=0.01,help='leaky_Relu')
    parser.add_argument('--nheads', type=int, default=1,help='nheads for self-attention')
    parser.add_argument('--data_friends_embedding', type=str, default="../data/Twitter/attention-embedding/embedding_length",
                        help='training set data dir')
    parser.add_argument('--walk_length', type=int, default=30,
                        help='length per walk')

    parser.add_argument('--joint', type=str, default='three_weight',
                        help='combination methods')
    parser.add_argument('--lamda', type=float, default=0.5,
                        help='grouping weight')
    args = parser.parse_args()

    if args.config != "no_file_exists":
        if os.path.exists(args.config):
            config = configparser.ConfigParser()
            config_file_path = args.config
            config.read(config_file_path)
            config_common = config['COMMON']
            for key in config_common.keys():
                args.__dict__[key] = config_common[key]
        else:
            print("config file named %s does not exist" % args.config)

    args.kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
    args.kernel_nums = [int(i) for i in args.kernel_nums.split(",")]

    #    # Check if args are valid
    #    assert args.rnn_size > 0, "rnn_size should be greater than 0"

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.model == "transformer":
        args.position = True
    else:
        args.position = False
    if args.debug.lower() == "true":
        args.debug = True
    else:
        args.debug = False

    if args.embedding_training.lower() == "true":
        args.embedding_training = True
    else:
        args.embedding_training = False
    if args.from_torchtext.lower() == "true":
        args.from_torchtext = True
    else:
        args.from_torchtext = False

    args.pretrain_embedding = args.pretrain_embedding.lower() == "true"
    args.pretrain_grouping = args.pretrain_grouping.lower() == "true"
    args.pretrain = args.pretrain.lower() == "true"
    args.group = args.group.lower() == "true"
    args.friend = args.friend.lower() == "true"

    args.device_id = args.gpu
    args.use_cuda = True

    return args
