import scipy.sparse as sp
import numpy as np

class MarkovChain():
  def __init__(self, item_dict, reversed_item_dict, item_freq_dict, transition_matrix):
    self.item_freq_dict = item_freq_dict
    self.item_dict = item_dict
    self.reversed_item_dict = reversed_item_dict
    self.nb_items = len(item_dict)
    # self.sp_matrix_path = sp_matrix_path
    self.transition_matrix = transition_matrix

  def top_predicted_item(self, previous_basket, topk):
    # candidate = np.zeros(self.nb_items)
    prev_basket_idx = [self.item_dict[item] for item in previous_basket]
    candidate = self.transition_matrix[prev_basket_idx].toarray().sum(axis=0)
    candidate = candidate / len(prev_basket_idx)
    topk_idx = np.argpartition(candidate, -topk)[-topk:]
    topk_item = [self.reversed_item_dict[item] for item in topk_idx]
    return topk_item