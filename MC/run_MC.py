import re
import numpy as np
import scipy.sparse as sp
from Utils import common_utils
import MarkovChain

def build_knowledge(training_instances):
    MAX_SEQ_LENGTH = 0
    item_freq_dict = {}
    user_dict = dict()

    for line in training_instances:
        elements = line.split("|")

        if len(elements) - 1 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 1

        user = elements[0]
        user_dict[user] = len(user_dict)

        basket_seq = elements[1:]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket.strip())
            for item_obs in item_list:
                if item_obs not in item_freq_dict:
                    item_freq_dict[item_obs] = 1
                else:
                    item_freq_dict[item_obs] += 1

    items = sorted(list(item_freq_dict.keys()))
    item_dict = dict()
    item_probs = []
    for item in items:
        item_dict[item] = len(item_dict)
        item_probs.append(item_freq_dict[item])

    item_probs = np.asarray(item_probs, dtype=np.float32)
    item_probs /= np.sum(item_probs)

    reversed_item_dict = dict(zip(item_dict.values(), item_dict.keys()))
    return MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict

def MC_hit_ratio(test_instances, topk, MC_model):
    hit_count = 0
    for line in test_instances:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[1:]
        last_basket = basket_seq[-1]
        prev_basket = basket_seq[-2]
        prev_item_idx = re.split('[\\s]+', prev_basket.strip())
        list_predict_item = MC_model.top_predicted_item(prev_item_idx, topk)
        item_list = re.split('[\\s]+', last_basket.strip())
        num_correct = len(set(item_list).intersection(list_predict_item))
        if num_correct > 0:
            hit_count += 1
    return hit_count / len(test_instances)


def MC_recall(test_instances, topk, MC_model):
    list_recall = []
    # total_correct = 0
    # total_user_correct = 0
    for line in test_instances:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[1:]
        last_basket = basket_seq[-1]
        prev_basket = basket_seq[-2]
        prev_item_idx = re.split('[\\s]+', prev_basket.strip())
        list_predict_item = MC_model.top_predicted_item(prev_item_idx, topk)
        item_list = re.split('[\\s]+', last_basket.strip())
        num_correct = len(set(item_list).intersection(list_predict_item))
        # total_correct += num_correct
        # if num_correct > 0:
        #   total_user_correct += 1
        list_recall.append(num_correct / len(item_list))
    return np.array(list_recall).mean()

if __name__ == '__main__':
    train_data_path = 'train_lines.txt'
    train_instances = common_utils.read_instances_lines_from_file(train_data_path)
    nb_train = len(train_instances)
    print(nb_train)

    test_data_path = 'test_lines.txt'
    test_instances = common_utils.read_instances_lines_from_file(test_data_path)
    nb_test = len(test_instances)
    print(nb_test)

    ### build knowledge ###
    # common_instances = train_instances + test_instances
    print("---------------------@Build knowledge-------------------------------")
    MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_freq_dict, user_dict = build_knowledge(train_instances)
    sp_matrix_path = 'transition_matrix_1.npz'
    mc_model = MarkovChain(item_dict, reversed_item_dict, item_freq_dict, sp_matrix_path)
    for topk in [5, 10, 20]:
        print("Top : ", topk)
        hit_rate = MC_hit_ratio(test_instances, topk, mc_model)
        recall = MC_recall(test_instances, topk, mc_model)
        print("hit ratio: ", hit_rate)
        print("recall: ", recall)