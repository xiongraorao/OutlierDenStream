#-*- coding: utf-8 -*-
"""

Created on 2019-02-22 17:11
@author Xiong Raorao

"""
from  sklearn import metrics


def cluster_eval(gt,pre):
    '''
    计算聚类的效果
    :param gt: 真实标签
    :param pre: 预测的结果
    :return:
    '''
    ret = {}
    ARI = metrics.adjusted_rand_score(gt, pre)
    AMI = metrics.adjusted_mutual_info_score(gt, pre)
    HCV = metrics.homogeneity_completeness_v_measure(gt, pre)
    FMI = metrics.fowlkes_mallows_score(gt, pre, True)
    ret['ARI'] = ARI
    ret['AMI'] = AMI
    ret['HCV'] = HCV
    ret['FMI'] = FMI
    return ret

def cluster_eval_withoutgt(X, labels):
    '''
    没有真实样本下的评估方法
    :param lables:
    :return:
    '''
    SC = metrics.silhouette_score(X, labels, metric='cosine')
    return SC