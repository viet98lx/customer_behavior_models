import scipy.sparse as sp

class MarkovChain():
  def __init__(self, item_dict, reversed_item_dict, item_freq_dict, sp_matrix_path):
    self.item_freq_dict = item_freq_dict
    self.item_dict = item_dict
    self.reversed_item_dict = reversed_item_dict
    self.nb_items = len(item_dict)
    self.sp_matrix_path = sp_matrix_path
    self.transition_matrix = sp.load_npz(self.sp_matrix_path)

  def top_predicted_item(self, previous_basket, topk):
    candidate = [0 for _ in range(self.nb_item)]
    prev_basket_idx = [self.item_dict[item] for item in previous_basket]
    for i in range(self.nb_items):
      score = 0
      for prev_idx in prev_basket_idx:
        score += self.transition_matrix[prev_idx, i]
      score = score / len(prev_basket_idx)
      candidate[i] = score
    topk_idx = np.argpartition(candidate, -topk)[-topk:]
    topk_item = [self.reversed_item_dict[item] for item in topk_idx]
    return topk_item