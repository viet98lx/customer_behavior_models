import itertools
import scipy.sparse as sp
import re

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
  transition_matrix = sp.csc_matrix((data, (row, col)), shape=(NB_ITEMS, NB_ITEMS), dtype="float32")
  nb_nonzero = len(pair_dict)
  density = nb_nonzero * 1.0 / NB_ITEMS / NB_ITEMS
  print("Density of first order matrix: {:.6f}".format(density))
  return transition_matrix

