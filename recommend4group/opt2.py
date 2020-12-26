import argparse, os
import configparser


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings

    parser.add_argument('--config', type=str, default="no_file_exists",
                        help='gpu number')
    parser.add_argument('--alias', type=str, default="concat",
                        help='gpu number')

    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden_dim')

    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='max_seq_len')
    parser.add_argument('--max_friend_len', type=int, default=400,
                        help='max_seq_len')
    parser.add_argument('--batch_size', type=int, default=256,
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

    parser.add_argument('--model', type=str, default="gmf",
                        help='model name')
    parser.add_argument('--friend', type=str, default="true",
                        help='need the following information')

    parser.add_argument('--data_train', type=str, default="../data/Weibo/train_for_tweet.csv",
                        help='training set data dir')
    parser.add_argument('--data_test', type=str, default="../data/Weibo/test_for_tweet.csv",
                        help='testing set data dir')
    parser.add_argument('--data_friends', type=str, default="../data/Weibo/profile.csv",
                        help='profile data dir')
    parser.add_argument('--type', type=str, default="tweet",
                        help='for user or for tweet')
    parser.add_argument('--position', type=bool, default=False,
                        help='gpu number')

    parser.add_argument('--keep_dropout', type=float, default=0.8,
                        help='keep_dropout')
    parser.add_argument('--l2', type=float, default=0,
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
    parser.add_argument('--group', type=str, default="True",
                        help='embedding group')

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
    parser.add_argument('--gpu', type=int, default=0,
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
    parser.add_argument('--data_grouping', type=str, default="../data/Weibo/friend_tag_for_tweet.csv",
                        help='get the information of grouping')
    parser.add_argument('--pretrain_grouping', type=str, default="false",
                        help='use the pretrain grouping weights or not')
    parser.add_argument('--learning_rate_ratio', type=float, default=0.1,
                        help='the lr ratio for grouping module compared with others')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='the lr ratio for grouping module compared with others')
    parser.add_argument('--l2_other', type=float, default=1e-5,
                        help='the l2 reg on model weights except the grouping module')

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

    args.pretrain_grouping = args.pretrain_grouping.lower() == "true"
    args.pretrain_embedding = args.pretrain_embedding.lower() == "true"
    args.pretrain = args.pretrain.lower() == "true"
    args.group = args.group.lower() == "true"
    args.friend = args.friend.lower() == "true"

    args.group_weight = '../data/Weibo/friend_tag_for_' + args.type + '_tfidf.npy'

    args.device_id = args.gpu
    args.use_cuda = True

    return args 
