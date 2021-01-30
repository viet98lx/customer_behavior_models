import itertools
import scipy.sparse as sp
import re
import numpy as np

def calculate_transition_matrix(train_instances, item_dict, item_freq_dict, reversed_item_dict):
  pair_dict = dict()
  NB_ITEMS = len(item_dict)
  for line in train_instances:
      elements = line.split("|")
      user = elements[0]
      basket_seq = elements[1:]
      for i in range(1,len(basket_seq)):
        prev_basket = basket_seq[i-1]
        cur_basket = basket_seq[i]
        prev_item_list = re.split('[\\s]+', prev_basket.strip())
        cur_item_list = re.split('[\\s]+', cur_basket.strip())
        prev_item_idx = [item_dict[item] for item in prev_item_list]
        cur_item_idx = [item_dict[item] for item in cur_item_list]
        for t in list(itertools.product(prev_item_idx, cur_item_idx)):
            if t in pair_dict.keys():
              pair_dict[t] += 1
            else:
              pair_dict[t] = 1

  for key in pair_dict.keys():
    pair_dict[key] /= item_freq_dict[reversed_item_dict[key[0]]]

  row = [p[0] for p in pair_dict]
  col = [p[1] for p in pair_dict]
  data = [pair_dict[p] for p in pair_dict]
  transition_matrix = sp.csr_matrix((data, (row, col)), shape=(NB_ITEMS, NB_ITEMS), dtype="float32")
  nb_nonzero = len(pair_dict)
  density = nb_nonzero * 1.0 / NB_ITEMS / NB_ITEMS
  print("Density of first order matrix: {:.6f}".format(density))
  return transition_matrix

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

def read_instances_lines_from_file(file_path):
    with open(file_path, "r") as f:
        lines = [line.rstrip('\n') for line in f]
        return lines

def FMC_hit_ratio(test_instances, topk, FMC_model):
    hit_count = 0
    # user_correct = set()
    # user_dict = dict()
    for line in test_instances:
        elements = line.split("|")
        # user = elements[0]
        # if user not in user_dict:
        #     user_dict[user] = len(user_dict)
        basket_seq = elements[1:]
        last_basket = basket_seq[-1]
        prev_basket = basket_seq[-2]
        prev_item_idx = re.split('[\\s]+', prev_basket.strip())
        list_predict_item = FMC_model.top_predicted_item(prev_item_idx, topk)
        item_list = re.split('[\\s]+', last_basket.strip())
        num_correct = len(set(item_list).intersection(list_predict_item))
        if num_correct > 0 :
            hit_count += 1
            # user_correct.add(user)
    return hit_count / len(test_instances)


def FMC_recall(test_instances, topk, FMC_model):
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
        list_predict_item = FMC_model.top_predicted_item(prev_item_idx, topk)
        item_list = re.split('[\\s]+', last_basket.strip())
        num_correct = len(set(item_list).intersection(list_predict_item))
        # total_correct += num_correct
        # if num_correct > 0:
        #   total_user_correct += 1
        list_recall.append(num_correct / len(item_list))
    return np.array(list_recall).mean()