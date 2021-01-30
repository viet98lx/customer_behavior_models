import torch
import torch.nn as nn
import numpy as np

class FPMC(nn.Module):
    def __init__(self, item_dict, user_dict, reversed_item_dict, n_factor):
        super(FPMC, self).__init__()
        self.item_dict = item_dict
        self.user_dict = user_dict
        self.reversed_item_dict = reversed_item_dict

        self.user_set = set(user_dict.values())
        self.item_set = set(item_dict.values())

        self.n_user = len(user_dict)
        self.n_item = len(item_dict)

        self.n_factor = n_factor
        # self.learn_rate = learn_rate
        # self.regular = regular

        self.VUI = nn.Parameter(torch.Tensor(self.n_user, self.n_factor))
        self.VIU = nn.Parameter(torch.Tensor(self.n_item, self.n_factor))
        self.VIL = nn.Parameter(torch.Tensor(self.n_item, self.n_factor))
        self.VLI = nn.Parameter(torch.Tensor(self.n_item, self.n_factor))

    def init_model(self, std=0.01):
        torch.nn.init.normal_(self.VUI, mean=0.0, std=std)
        torch.nn.init.normal_(self.VIU, mean=0.0, std=std)
        torch.nn.init.normal_(self.VIL, mean=0.0, std=std)
        torch.nn.init.normal_(self.VLI, mean=0.0, std=std)
        # self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        # self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))

        # self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        # self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))

    def compute_x(self, u, i, b_tm1):
        # acc_val = 0.0
        # for l in b_tm1:
        #     acc_val += np.dot(self.VIL[i], self.VLI[l])
        acc_ = torch.matmul(self.VIL[i], self.VLI[b_tm1].t())
        acc_val = torch.mean(acc_, dim=0)
        # return (np.dot(self.VUI[u], self.VIU[i]) + (acc_val / len(b_tm1)))
        return torch.dot(self.VUI[u], self.VIU[i]) + acc_val

    def predict_next_item_score(self, u, b_tm1):
        rank_score = torch.matmul(self.VUI[u], self.VIU.t()) + torch.mean(torch.matmul(self.VIL, self.VLI[b_tm1].t()),
                                                                          dim=1)
        return rank_score

    def top_k_recommendations(self, sample, topk=10):
        ''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
        '''
        u, b_tm1, target_basket = sample[0], sample[1], sample[2]
        # print(u)
        # print(b_tm1)
        # print(target_basket)
        score = self.predict_next_item_score(u, b_tm1)
        # idx = list(np.argpartition(score, -topk)[-topk:])
        idx = torch.topk(score, k=topk).indices.tolist()

        # find top k according to output
        # return np.mean(np.array(list_recall), axis=0)
        return idx

    def compute_recall(self, data_list, topk=10):
        list_recall = []
        for u, b_tm1, target_basket in data_list:
            idx = self.top_k_recommendations((u, b_tm1, target_basket), topk)
            correct = len(set(idx).intersection(set(target_basket)))
            list_recall.append(correct / len(target_basket))
        # find top k according to output
        return np.mean(np.array(list_recall), axis=0)

    def compute_hitrate(self, data_list, topk=10):
        # list_hitrate = []
        hit_rate = 0
        for u, b_tm1, target_basket in data_list:
            idx = self.top_k_recommendations((u, b_tm1, target_basket), topk)
            correct = len(set(idx).intersection(set(target_basket)))
            # list_hitrate.append(correct / len(target_basket))
            if correct > 0:
                hit_rate += 1
        # find top k according to output
        return hit_rate / len(data_list)

    def evaluation(self, data_list, topk):
        recall = self.compute_recall(data_list, topk)
        return recall

    def forward(self, u, i, j, b_tm1):
        z1 = self.compute_x(u, i, b_tm1)
        z2 = self.compute_x(u, j, b_tm1)
        loss = 1 - torch.sigmoid(z1 - z2)
        # print(loss)
        return loss