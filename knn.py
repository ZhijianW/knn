# coding=utf-8
from __future__ import division
from numpy import *
from operator import itemgetter
import matplotlib.pyplot as plt


def classify(input_mat, data_set):
    data_set_size = data_set.shape[0]
    # KNN的算法核心就是欧式距离的计算，一下三行是计算待分类的点和训练集中的任一点的欧式距离
    diff_mat = tile(input_mat, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat**2
    distance = sq_diff_mat.sum(axis=1)**0.5
    # 接下来是一些统计工作
    distance.sort()
    return distance[0]


def file2mat(test_filename, para_num):
    fr = open(test_filename)
    lines = fr.readlines()
    line_nums = len(lines)
    result_mat = zeros((line_nums, para_num))
    class_label = []
    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = line.split(',')
        result_mat[i, :] = item_mat[0: para_num]
        class_label.append(item_mat[-1])
    fr.close()
    return result_mat


# 为了防止某个属性对结果产生很大的影响，所以有了这个优化，比如:10000,4.5,6.8 10000就对结果基本起了决定作用
def auto_norm(data_set):
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_val
    norm_mat = zeros(shape(data_set))
    mat_size = norm_mat.shape[0]
    norm_mat = data_set - tile(min_val, (mat_size, 1))
    norm_mat = norm_mat / tile(ranges, (mat_size, 1))
    return norm_mat, min_val, ranges


def roc(data_set):
    normal = 0
    data_set_size = data_set.shape[0]
    roc_rate = zeros((data_set_size, 2))
    for i in range(data_set_size):
        if data_set[i, 1:2] == 1:
            normal += 1
    abnormal = data_set_size - normal
    sorted(data_set, key=itemgetter(0), reverse=True)
    for j in range(1000):
        threshold = data_set[-1][0] / 1000 * j
        normal1 = 0
        abnormal1 = 0
        for k in range(data_set_size):
            if data_set[k, 0:1] > threshold and data_set[k, 1:2] == 1:
                normal1 += 1
            if data_set[k, 0:1] > threshold and data_set[k, 1:2] == 2:
                abnormal1 += 1
        roc_rate[j, 0:1] = normal1 / normal
        roc_rate[j, 1:2] = abnormal1 / abnormal
        plt.plot(roc_rate[j, 0:1], roc_rate[j, 1:2], 'b+')
    plt.show()
    return 0


def test(training_filename, test_filename):
    training_mat, class_label = file2mat(training_filename, 32)
    # training_mat, min_val, ranges = auto_norm(training_mat)
    test_mat, test_label = file2mat(test_filename, 32)
    test_size = test_mat.shape[0]
    result = zeros((test_size, 3))
    for i in range(test_size):
        result[i] = (i + 1, classify(test_mat[i], training_mat), test_label[i])
		if result[i, 1:2] == 2:
            plt.plot(i, result[i, 0:1], 'r+')
	    else:
            plt.plot(i, result[i, 0:1], 'b+')
    plt.show()
    roc(result)
    return 0

if __name__ == "__main__":
    test('1.txt', '2.txt')
