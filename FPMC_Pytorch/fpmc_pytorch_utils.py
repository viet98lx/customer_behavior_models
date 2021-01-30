import sys, os, pickle, time
import math, random
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv, math
import re

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
    # train_instances = common_utils.read_instances_lines_from_file(train_data_path)
    # test_instances = common_utils.read_instances_lines_from_file(test_data_path)
    train_instances = read_instances_lines_from_file(train_data_path)
    test_instances = read_instances_lines_from_file(test_data_path)
    return train_instances, test_instances


def data_to_3_list(data_instances, item_dict, user_dict, reversed_item_dict):
    data_list = []
    for line in data_instances:
        elements = line.split('|')
        user = elements[0]
        user_idx = user_dict[user]
        prev_basket = [item_dict[item] for item in re.split('[\\s]+', elements[1].strip())]
        target_basket = [item_dict[item] for item in re.split('[\\s]+', elements[-1].strip())]
        data_list.append((user_idx, prev_basket, target_basket))

    return data_list

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(init_model, load_path):
    init_model.load_state_dict(torch.load(load_path))
    return init_model

def learn_epoch(model, optimizer, tr_data, neg_batch_size):
    tracking_loss = []
    for iter_idx in range(len(tr_data)):
        # (u, b_tm1, target_basket) = random.choice(tr_data)
        avg_loss = 0
        model.zero_grad()
        (u, b_tm1, target_basket) = tr_data[iter_idx]
        exclu_set = model.item_set - set(target_basket)
        j_list = random.sample(exclu_set, neg_batch_size)
        i = target_basket[0]
        for j in j_list:
            loss = model(u, i, j, b_tm1)
            # avg_loss += loss
            # loss.backward()
            # optimizer.step()
            # loss_iter += loss.detach().item()
            # print(loss)
            avg_loss += loss
        mean_loss = avg_loss / neg_batch_size
        mean_loss.backward()
        optimizer.step()
        tracking_loss.append(mean_loss.detach().item())
    return np.array(tracking_loss).mean()

def learnSBPR_FPMC(model, model_name, optimizer, tr_data, te_data=None, n_epoch=10, neg_batch_size=10, eval_per_epoch=False):
    max_recall = 0
    losses = []
    for epoch in range(n_epoch):
        print("start epoch: ", epoch)
        shuffle(tr_data)
        loss_ep = learn_epoch(model, optimizer, tr_data, neg_batch_size=neg_batch_size)
        losses.append(loss_ep)
        if eval_per_epoch == True:
            with torch.no_grad():
              model.eval()
              recall_test = model.evaluation(te_data, 10)
              recall_train = model.evaluation(tr_data, 10)
              print("Recall train: ", recall_train)
              print("Recall test: ", recall_test)
              save_model(model,model_name+'_best_model.pt')

            if (recall_test > max_recall):
              print("Recall increase from %.6f to %.6f" % (max_recall, recall_test))
              max_recall = recall_test
              # filename = 'best_epoch_'+str(epoch)+'.npz'
              # self.save(filename)
            print('epoch %d done' % epoch)
        else:
            print('epoch %d done' % epoch)
    ep = range(1, len(losses)+1)
    plt.plot(ep,losses, 'b-')
    plt.savefig('Loss.png')
    plt.show()