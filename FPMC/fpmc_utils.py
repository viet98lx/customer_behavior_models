
import math, random
from random import shuffle
import numpy as np
# from Utils import common_utils
import re


def sigmoid(x):
    x = min(10, max(-10, x))
    return 1 / (1 + math.exp(-x))


def build_knowledge(training_instances):
    MAX_SEQ_LENGTH = 0
    item_freq_dict = {}
    user_dict = dict()

    for line in training_instances:
        elements = line.split("|")

        if len(elements) - 1 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 1

        user = elements[0]
        if user not in user_dict:
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

def read_instances_lines_from_file(file_path):
    with open(file_path, "r") as f:
        lines = [line.rstrip('\n') for line in f]
        return lines

def load_data_from_dir(dirname):
    train_data_path = dirname + '/' + 'train_lines.txt'
    test_data_path = dirname + '/' + 'test_lines.txt'
    train_instances = read_instances_lines_from_file(train_data_path)
    test_instances = read_instances_lines_from_file(test_data_path)
    # train_instances = read_instances_lines_from_file(train_data_path)
    # test_instances = read_instances_lines_from_file(test_data_path)
    return train_instances, test_instances


def data_to_3_list(data_instances, item_dict, user_dict, reversed_item_dict):
    data_list = []
    for line in data_instances:
        elements = line.split('|')
        # print(item_dict)
        # print(elements)
        user = elements[0]
        user_idx = user_dict[user]
        prev_basket = [item_dict[item] for item in re.split('[\\s]+', elements[1].strip())]
        target_basket = [item_dict[item] for item in re.split('[\\s]+', elements[-1].strip())]
        data_list.append((user_idx, prev_basket, target_basket))

    return data_list