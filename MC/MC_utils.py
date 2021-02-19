import itertools
import scipy.sparse as sp
import re
import numpy as np

def calculate_transition_matrix(train_instances, item_dict, item_freq_dict, reversed_item_dict, mc_order):
  pair_dict = dict()
  NB_ITEMS = len(item_dict)
  for line in train_instances:
      elements = line.split("|")
      user = elements[0]
      basket_seq = elements[1:]
      st = mc_order
      if len(basket_seq) < mc_order + 1:
          st = 1
      for i in range(st, len(basket_seq)):
          prev_baskets = basket_seq[i - st:i]
          cur_basket = basket_seq[i]
          prev_item_list = []
          for basket in prev_baskets:
            prev_item_list += re.split('[\\s]+', basket.strip())
          cur_item_list = re.split('[\\s]+', cur_basket.strip())
          prev_item_idx = [item_dict[item] for item in prev_item_list]
          cur_item_idx = [item_dict[item] for item in cur_item_list]
          for t in list(itertools.product(prev_item_idx, cur_item_idx)):
            if t in pair_dict:
                pair_dict[t] += 1
            else:
                pair_dict[t] = 1

  for key in pair_dict:
    pair_dict[key] /= item_freq_dict[reversed_item_dict[key[0]]]

  row = [p[0] for p in pair_dict]
  col = [p[1] for p in pair_dict]
  data = [pair_dict[p] for p in pair_dict]
  transition_matrix = sp.csr_matrix((data, (row, col)), shape=(NB_ITEMS, NB_ITEMS), dtype="np.float16")
  nb_nonzero = len(pair_dict)
  density = nb_nonzero * 1.0 / NB_ITEMS / NB_ITEMS
  print("Density of matrix: {:.6f}".format(density))
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

def write_predict(file_name, test_instances, topk, MC_model):
    f = open(file_name, 'w')
    for line in test_instances:
        elements = line.split("|")
        user = elements[0]
        basket_seq = elements[-MC_model.mc_order-1:-1]
        last_basket = basket_seq[-1]
        # prev_basket = basket_seq[-2]
        prev_item_list = []
        for basket in basket_seq:
            prev_item_list += [p for p in re.split('[\\s]+', basket.strip())]
        list_predict_item = MC_model.top_predicted_item(prev_item_list, topk)
        # item_list = re.split('[\\s]+', last_basket.strip())
        cur_item_list = [p for p in re.split('[\\s]+', last_basket.strip())]
        f.write(str(user)+'\n')
        f.write('ground_truth:')
        for item in cur_item_list:
            f.write(' '+str(item))
        f.write('\n')
        f.write('predicted:')
        predict_len = len(list_predict_item)
        for i in range(predict_len):
            f.write(' '+str(list_predict_item[predict_len-1-i]))
        f.write('\n')
    f.close()

def read_predict(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    list_ground_truth_basket = []
    list_predict_basket = []
    for i in range(0, len(lines), 3):
        user = lines[i].strip('\n')
        list_ground_truth_basket.append(re.split('[\\s]+',lines[i+1].strip('\n'))[1:])
        list_predict_basket.append(re.split('[\\s]+',lines[i+2].strip('\n'))[1:])

    return list_ground_truth_basket, list_predict_basket

def hit_ratio(list_ground_truth_basket, list_predict_basket, topk):
    hit_count = 0
    for gt, predict in zip(list_ground_truth_basket, list_predict_basket):
        num_correct = len(set(gt).intersection(predict[:topk]))
        if num_correct > 0:
            hit_count += 1
            # user_correct.add(user)
    return hit_count / len(list_ground_truth_basket)

def recall(list_ground_truth_basket, list_predict_basket, topk):
    list_recall = []
    for gt, predict in zip(list_ground_truth_basket, list_predict_basket):
        num_correct = len(set(gt).intersection(predict[:topk]))
        list_recall.append(num_correct / len(gt))
    return np.array(list_recall).mean()