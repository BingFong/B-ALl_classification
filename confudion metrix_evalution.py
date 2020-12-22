# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 02:21:45 2020

@author: CILAB
"""
from glob import glob
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

def evaluate(matrix_e):     # evaluate terms from matrix_e
    terms = ['Accuracy', 'F1-score', 'Sensitivity', 'Precision', 'Specificity'
             , 'Recall']    #terms of cofusion matrix
    count = 0
    for term in terms:

        min = np.min(matrix_e[:, count])
        mean = np.mean(matrix_e[:, count])
        std = np.std(matrix_e[:, count])
        max = np.max(matrix_e[:, count])

        print(term)
        print("{0:^6s} {1:^6s} +- {2:^6s} {3:^6s}".format('Min.', 'Mean', 'Std', 'Max.'))
        print("{0:^.4f} {1:^.4f} +- {2:^.4f} {3:^.4f}\n".format(min, mean, std, max))

        count += 1

def get_data(file, terms):      #get result datas from result.npz
    result = np.load(file)
    f1 = result['f1'][1:]
    tar = np.argmax(f1) + 1    #target index of terms from result

    cm = np.zeros(len(terms))    #confusion metrix
    index = 0    #index of rows
    for term in terms:
        cm[index] = result[term][tar]
        index += 1
    return f1, cm


if __name__ == '__main__':
    dataroot = r'results/exp_4'
    terms = ['acc', 'f1', 'acc_all', 'precision', 'acc_hem', 'acc_all']    #terms of cofusion matrix
    matrix_e = np.zeros(len(terms))    #load all terms from all results
    result_count = 0    #results counter

    for file in glob(join(dataroot, '*', 'results.npz')):
        result_count += 1
        f1, cm = get_data(file, terms)
        matrix_e = np.vstack((matrix_e, cm))
        
        plt.plot(range(1, f1.shape[0]+1), f1[:],)# label=result_count)
        
#        if result_count > 4:
#            break

    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('F1-score comparison')
    plt.legend()
    plt.show()

    matrix_e = np.delete(matrix_e, (0), axis=0)    #delete empty row
    evaluate(matrix_e)

    print('f1-score:', matrix_e[:, 1])
    print('實驗次數：', result_count)
    