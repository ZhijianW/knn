# coding=utf-8
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def classify(input_vct, data_set):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(input_vct, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat**2
    distance = sq_diff_mat.sum(axis=1)**0.5
    return distance.min(axis=0)


def file2mat(test_filename, para_num):
    fr = open(test_filename)
    lines = fr.readlines()
    line_nums = len(lines)
    result_mat = np.zeros((line_nums, para_num))
    class_label = []
    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = line.split(',')
        result_mat[i, :] = item_mat[0: para_num]
        class_label.append(item_mat[-1])
    fr.close()
    return result_mat, class_label


def roc(data_set):
    normal = 0
    data_set_size = data_set.shape[1]
    roc_rate = np.zeros((2, data_set_size))
    for i in range(data_set_size):
        if data_set[2][i] == 1:
            normal += 1
    abnormal = data_set_size - normal
    max_dis = data_set[1].max()
    for j in range(1000):
        threshold = max_dis / 1000 * j
        normal1 = 0
        abnormal1 = 0
        for k in range(data_set_size):
            if data_set[1][k] > threshold and data_set[2][k] == 1:
                normal1 += 1
            if data_set[1][k] > threshold and data_set[2][k] == 2:
                abnormal1 += 1
        roc_rate[0][j] = normal1 / normal
        roc_rate[1][j] = abnormal1 / abnormal
    return roc_rate


def test(training_filename, test_filename):
    training_mat, training_label = file2mat(training_filename, 32)
    test_mat, test_label = file2mat(test_filename, 32)
    test_size = test_mat.shape[0]
    result = np.zeros((test_size, 3))
    for i in range(test_size):
        result[i] = i + 1, classify(test_mat[i], training_mat), test_label[i]
    result = np.transpose(result)
    plt.figure(1)
    plt.scatter(result[0], result[1], facecolors=result[2], edgecolors='None', s=1, alpha=1)
    roc_rate = roc(result)
    plt.figure(2)
    plt.scatter(roc_rate[0], roc_rate[1], edgecolors='None', s=1, alpha=1)


if __name__ == "__main__":
    test('1.txt', '2.txt')