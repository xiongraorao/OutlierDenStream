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
    ARI = metrics.adjusted_rand_score(gt, pre) # 调整的Rand Index， 和RI相比，实现了随机结果趋近于0
    AMI = metrics.adjusted_mutual_info_score(gt, pre) # 调整的互信息量
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
    ret = {}
    SC = metrics.silhouette_score(X, labels, metric='cosine')
    CH = metrics.calinski_harabaz_score(X, labels)
    DBI = metrics.davies_bouldin_score(X, labels) # 越小越好
    ret['SC'] = SC
    ret['CH'] = CH
    ret['DBI'] = DBI
    return ret

def trans_sqlinsert(x):
    '''
    把二维的list, tuple 转换成sql中的values 后面的东西
    :param x:
    :return:
    '''
    if x is None or len(x) == 0:
        return None
    elif len(x) == 1:
        x = tuple(map(lambda a:tuple(a), x))
        return str(tuple(x))[1:-2]
    else:
        x = tuple(map(lambda a:tuple(a), x))
        return str(tuple(x))[1:-1]