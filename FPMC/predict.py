import sys, os, pickle, argparse
import re
import fpmc_utils
import fpmc
import random
# sys.path.append(os.path.abspath(os.path.join('..', 'data')))
sys.path.append('..')


# from FPMC import FPMC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='fpmc')
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=10)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=10)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=16)
    parser.add_argument('-l', '--lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.001)
    parser.add_argument('--nb_predict', help='# of predict', type=int, default=10)
    parser.add_argument('--topk', help='# of predict', type=int, default=10)
    args = parser.parse_args()

    f_dir = args.input_dir
    epoch = args.n_epoch
    n_neg = args.n_neg
    n_factor = args.n_factor
    learn_rate = args.lr
    regular = args.regular
    o_dir = args.output_dir
    model_name = args.model_name
    nb_predict = args.nb_predict
    topk = args.topk

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
    load_file = os.path.join(o_dir, model_name)
    fpmc.load(load_file)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    for i in random.sample(test_data_list, nb_predict):
        (u, b_tm1, target_basket) = i
        idx = fpmc.top_k_recommendations(i, topk)
        topk_item = [reversed_item_dict[i] for i in idx]
        prev_item = [reversed_item_dict[i] for i in b_tm1]
        target_item = [reversed_item_dict[i] for i in target_basket]
        correct_set = set(topk_item).intersection(set(target_item))
        print("Input basket: ", prev_item)
        print("Ground truth: ", target_item)
        print("Nb_correct: ", len(correct_set))
        print("Predict topk: ", topk_item)
        print("Items correct: ", list(correct_set))
        print()
