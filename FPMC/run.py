import sys, os, pickle, argparse
import re
import fpmc_utils
import fpmc
sys.path.append(os.path.abspath(os.path.join('..', 'data')))


# from FPMC import FPMC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='data/')
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=10)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=10)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=16)
    parser.add_argument('-l', '--lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.001)
    args = parser.parse_args()

    f_dir = args.input_dir
    epoch = args.n_epoch
    n_neg = args.n_neg
    n_factor = args.n_factor
    learn_rate = args.lr
    regular = args.regular

    data_dir = f_dir
    train_instances, test_instances = fpmc_utils.load_data_from_dir(data_dir)
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = fpmc_utils.build_knowledge(train_instances + test_instances)

    train_data_list = fpmc_utils.data_to_3_list(train_instances, item_dict, user_dict, reversed_item_dict)
    # shuffle(train_data_list)
    test_data_list = fpmc_utils.data_to_3_list(test_instances, item_dict, user_dict, reversed_item_dict)

    # train_ratio = 0.8
    # split_idx = int(len(data_list) * train_ratio)
    # tr_data = data_list[:split_idx]
    # te_data = data_list[split_idx:]
    fpmc = fpmc.FPMC(item_dict = item_dict, user_dict = user_dict, reversed_item_dict = reversed_item_dict,
                n_factor= n_factor, learn_rate=learn_rate, regular=regular)
    # fpmc.user_set = user_set
    # fpmc.item_set = item_set
    fpmc.init_model()
    fpmc.learnSBPR_FPMC(train_data_list, test_data_list, n_epoch=epoch,
                                    neg_batch_size=n_neg, eval_per_epoch=True)