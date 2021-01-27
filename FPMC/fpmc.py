import csv, math
import numpy as np
import sys, os, pickle, time
import math, random
from random import shuffle
import numpy as np
import fpmc_utils
import re


class FPMC():
    def __init__(self, item_dict, user_dict, reversed_item_dict, n_factor, learn_rate, regular):
        self.item_dict = item_dict
        self.user_dict = user_dict
        self.reversed_item_dict = reversed_item_dict

        self.user_set = set(user_dict.values())
        self.item_set = set(item_dict.values())

        self.n_user = len(user_dict)
        self.n_item = len(item_dict)

        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

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
        np.savez(filename, V_user_item=self.VUI, V_item_user=self.VIU, V_prev_next=self.VLI, V_next_prev=self.VIL)

    def load(self, filename):
        '''Load parameters values form a file
        '''
        f = np.load(filename)
        self.VUI = f['V_user_item']
        self.VIU = f['V_item_user']
        self.VLI = f['V_prev_next']
        self.VIL = f['V_next_prev']

    def init_model(self, std=0.01):
        self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))

        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))

    def compute_x(self, u, i, b_tm1):
        acc_val = 0.0
        # for l in b_tm1:
        #     acc_val += np.dot(self.VIL[i], self.VLI[l])
        acc_val = np.mean(np.dot(self.VIL[i], self.VLI[b_tm1].T), axis=0)
        # return (np.dot(self.VUI[u], self.VIU[i]) + (acc_val / len(b_tm1)))
        return (np.dot(self.VUI[u], self.VIU[i]) + (acc_val))

    def predict_next_item_score(self, u, b_tm1):
        rank_score = np.dot(self.VUI[u], self.VIU.T) + np.mean(np.dot(self.VIL, self.VLI[b_tm1].T), axis=1)
        return rank_score

    def top_k_recommendations(self, sample, topk=10):
        ''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
        '''
        u, b_tm1, target_basket = sample[0], sample[1], sample[2]
        # print(u)
        # print(b_tm1)
        # print(target_basket)
        score = self.predict_next_item_score(u, b_tm1)
        idx = list(np.argpartition(score, -topk)[-topk:])

        # find top k according to output
        # return np.mean(np.array(list_recall), axis=0)
        return idx

    def compute_recall(self, data_list, topk=10):
        list_recall = []
        for u, b_tm1, target_basket in data_list:
            idx = self.top_k_recommendations((u, b_tm1, target_basket), topk)
            correct_set = set(idx).intersection(set(target_basket))
            # for item in correct_set:
            #     print(self.reversed_item_dict[item], ' ')
            correct = len(correct_set)
            # print()
            list_recall.append(correct / len(target_basket))
        # find top k according to output
        return np.mean(np.array(list_recall), axis=0)

    def evaluation(self, data_list, topk):
        recall = self.compute_recall(data_list, topk)
        return recall

    def learn_epoch(self, tr_data, neg_batch_size):
        for iter_idx in range(len(tr_data)):
            (u, b_tm1, target_basket) = random.choice(tr_data)
            # (u, b_tm1, target_basket) = tr_data[iter_idx]

            exclu_set = self.item_set - set(target_basket)
            j_list = random.sample(exclu_set, neg_batch_size)
            i = target_basket[0]
            z1 = self.compute_x(u, i, b_tm1)
            for j in j_list:
                z2 = self.compute_x(u, j, b_tm1)
                delta = 1 - fpmc_utils.sigmoid(z1 - z2)

                VUI_update = self.learn_rate * (delta * (self.VIU[i] - self.VIU[j]) - self.regular * self.VUI[u])
                VIUi_update = self.learn_rate * (delta * self.VUI[u] - self.regular * self.VIU[i])
                VIUj_update = self.learn_rate * (-delta * self.VUI[u] - self.regular * self.VIU[j])

                self.VUI[u] += VUI_update
                self.VIU[i] += VIUi_update
                self.VIU[j] += VIUj_update

                eta = np.mean(self.VLI[b_tm1], axis=0)
                VILi_update = self.learn_rate * (delta * eta - self.regular * self.VIL[i])
                VILj_update = self.learn_rate * (-delta * eta - self.regular * self.VIL[j])
                VLI_update = self.learn_rate * (
                        (delta * (self.VIL[i] - self.VIL[j]) / len(b_tm1)) - self.regular * self.VLI[b_tm1])

                self.VIL[i] += VILi_update
                self.VIL[j] += VILj_update
                self.VLI[b_tm1] += VLI_update

    def learnSBPR_FPMC(self, tr_data, output_dir, model_name, te_data=None, n_epoch=10, neg_batch_size=10, eval_per_epoch=False):
        max_recall = 0
        for epoch in range(n_epoch):
            print('Start epoch: ', epoch)
            shuffle(tr_data)
            self.learn_epoch(tr_data, neg_batch_size=neg_batch_size)
            if eval_per_epoch == True:
                recall_topk = self.evaluation(te_data, 10)
                if (recall_topk > max_recall):
                    print("Recall increase from %.6f to %.6f" % (max_recall, recall_topk))
                    max_recall = recall_topk
                    filename = output_dir + model_name + '_best_epoch_' + str(epoch) + '.npz'
                    self.save(filename)
                print('epoch %d done' % epoch)
            else:
                print('epoch %d done' % epoch)
