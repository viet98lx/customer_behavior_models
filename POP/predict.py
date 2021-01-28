import sys, os, pickle, argparse
import re
import random
# sys.path.append(os.path.abspath(os.path.join('..', 'data')))
import utils
import numpy as np
from POP import POP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='../data/')
    parser.add_argument('--output_dir', help='The directory of output', type=str, default='../saved_models/')
    parser.add_argument('--model_name', help='Model name ', type=str, default='fpmc')
    parser.add_argument('--nb_predict', help='# of predict', type=int, default=10)
    parser.add_argument('--topk', help='# of predict', type=int, default=10)
    parser.add_argument('--example_file', help='Example_file', type=str, default=None)
    args = parser.parse_args()

    f_dir = args.input_dir
    o_dir = args.output_dir
    model_name = args.model_name
    nb_predict = args.nb_predict
    topk = args.topk
    ex_file = args.example_file

    data_dir = f_dir
    train_data_path = data_dir + 'train_lines.txt'
    train_instances = utils.read_instances_lines_from_file(train_data_path)
    nb_train = len(train_instances)
    # print(nb_train)

    test_data_path = data_dir + 'test_lines.txt'
    test_instances = utils.read_instances_lines_from_file(test_data_path)
    nb_test = len(test_instances)
    # print(nb_test)
    # print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = utils.build_knowledge(train_instances + test_instances)

    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    saved_file = os.path.join(o_dir, model_name)
    # print("Save model in ", saved_file)
    f = np.load(saved_file, item_probs)
    item_probs = f['item_probs']
    # print(item_probs)
    pop_model = POP(item_dict, reversed_item_dict, item_probs)

    if ex_file is not None:
        ex_instances = utils.read_instances_lines_from_file(ex_file)
    else :
        ex_instances = test_instances
    for i in random.sample(ex_instances, nb_predict):
        basket = i.split('|')[2]
        target_basket = [item for item in re.split('[\\s]+',basket.strip())]
        topk_item = pop_model.top_popular_item(topk)
        correct_set = set(topk_item).intersection(set(target_basket))
        print("Ground truth: ", target_basket)
        print("Nb_correct: ", len(correct_set))
        print("Predict topk: ", topk_item)
        print("Items correct: ", list(correct_set))
        print()