import pickle
import numpy as np
class FMC():
    def __init__(self, item_dict, user_dict, reversed_item_dict, n_factor):
        self.item_dict = item_dict
        self.user_dict = user_dict
        self.reversed_item_dict = reversed_item_dict

        self.user_set = set(user_dict.values())
        self.item_set = set(item_dict.values())

        self.n_user = len(user_dict)
        self.n_item = len(item_dict)

        self.n_factor = n_factor
        self.W = None
        self.H = None

    @staticmethod
    def dump(fpmcObj, fname):
        pickle.dump(fpmcObj, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, 'rb'))

    def save(self, filename):
        '''Save the parameters of a network into a file
        '''
        print('Save model in ' + filename)
        # if not os.path.exists(os.path.dirname(filename)):
        #   os.makedirs(os.path.dirname(filename))
        np.savez(filename, W=self.W, H=self.H)

    def load(self, filename):
        '''Load parameters values form a file
        '''
        f = np.load(filename)
        self.W = f['W']
        print(self.W.shape)
        self.H = f['H']
        print(self.H.shape)

    def top_predicted_item(self, previous_basket, topk):
        candidate = np.zeros(self.n_item)
        prev_basket_idx = [self.item_dict[item] for item in previous_basket]
        # for item_idx in prev_basket_idx:
        for i in range(self.n_item):
            candidate[i] = np.matmul(self.W[prev_basket_idx,:], self.H[:,i]).mean(axis=0)
        topk_idx = np.argpartition(candidate, -topk)[-topk:]
        topk_item = [self.reversed_item_dict[item] for item in topk_idx]
        # print("Done")
        return topk_item
    # def init_model(self, std=0.01):
    #     self.W = np.random.normal(0, std, size=(self.n_item, self.n_factor))
    #     self.H = np.random.normal(0, std, size=(self.n_item, self.n_factor))
