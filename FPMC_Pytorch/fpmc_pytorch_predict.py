import sys, os, pickle, argparse
import re
import fpmc_pytorch_utils
import fpmc_pytorch
import torch
import random
# sys.path.append(os.path.abspath(os.path.join('..', 'data')))
sys.path.append('..')


# from FPMC import FPMC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='fpmc')
    # parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=10)
    # parser.add_argument('--n_neg', help='# of neg samples', type=int, default=10)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=16)
    # parser.add_argument('-l', '--lr', help='learning rate', type=float, default=0.01)
    # parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.001)
    parser.add_argument('--nb_predict', help='# of predict', type=int, default=10)
    parser.add_argument('--topk', help='# of predict', type=int, default=10)
    parser.add_argument('--example_file', help='Example_file', type=str, default=None)
    args = parser.parse_args()

    f_dir = args.input_dir
    # epoch = args.n_epoch
    # n_neg = args.n_neg
    n_factor = args.n_factor
    # learn_rate = args.lr
    # regular = args.regular
    o_dir = args.output_dir
    model_name = args.model_name
    nb_predict = args.nb_predict
    topk = args.topk
    ex_file = args.example_file

    data_dir = f_dir
    train_instances, test_instances = fpmc_pytorch_utils.load_data_from_dir(data_dir)

    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = fpmc_pytorch_utils.build_knowledge(train_instances + test_instances)

    train_data_list = fpmc_pytorch_utils.data_to_3_list(train_instances, item_dict, user_dict, reversed_item_dict)
    # shuffle(train_data_list)
    test_data_list = fpmc_pytorch_utils.data_to_3_list(test_instances, item_dict, user_dict, reversed_item_dict)

    init_fpmc = fpmc_pytorch.FPMC(item_dict=item_dict, user_dict=user_dict, reversed_item_dict=reversed_item_dict,
                n_factor=n_factor)
    init_fpmc.init_model()
    fpmc = fpmc_pytorch_utils.load_model(init_fpmc, o_dir+model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    fpmc.to(device)
    # fpmc.init_model()
    # optimizer = torch.optim.Adam(fpmc.parameters(), lr=learn_rate)
    fpmc.eval()
    if ex_file is not None:
        ex_instances = fpmc_pytorch_utils.read_instances_lines_from_file(ex_file)
    else :
        ex_instances = test_instances
    recall_test = fpmc.compute_recall(test_data_list, topk=10)
    hit_rate = fpmc.compute_hitrate(test_data_list, topk=10)
    print("Recall test: %.4f" % (recall_test))
    print("Hit ratio test: %.4f" % (hit_rate))
    for i in random.sample(ex_instances, nb_predict):
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