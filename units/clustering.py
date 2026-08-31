import numpy as np
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
import sys
from munkres import Munkres
'''The clustering isn't importance, but representation learning'''
'''
函数名	作用
get_score	计算当前表示的聚类指标，并存入列表
Clustering	执行 KMeans 聚类，并计算评估指标
calculate_cost_matrix	计算聚类标签与真实标签匹配的代价矩阵
get_cluster_labels_from_indices	根据匈牙利算法的最优匹配，返回调整后的标签
get_y_preds	通过匈牙利算法优化 KMeans 预测的标签，使其与真实标签对齐
classification_metric	计算分类指标（ACC、Precision、Recall、F-score）
clustering_metric	计算聚类指标（ACC、NMI、ARI、AMI）
get_cluster_sols	使用 KMeans 进行聚类，并返回聚类标签

'''
def get_score(representation, Y_list, acc, nmi, ARI, f_mea):
    # get clustering results and append them to list
    y_preds, scores = Clustering(representation, Y_list[0]) # 调用聚类函数
    acc.append(scores['kmeans']['accuracy'])
    nmi.append(scores['kmeans']['NMI'])
    f_mea.append(scores['kmeans']['f_measure'])
    ARI.append(scores['kmeans']['ARI'])

    return scores


def Clustering(x_list, y):
    n_clusters = np.size(np.unique(y))  # 聚类的类别数 = 标签的个数。

    x_final_concat = np.concatenate(x_list[:], axis=1)
    kmeans_assignments, km = get_cluster_sols(x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10}) # 进行KMEANS聚类，返回的是kmeans聚类的标签
    y_preds = get_y_preds(y, kmeans_assignments, n_clusters)    # 重新排列标签，使其与真是标签y对齐
    if np.min(y) == 1:
        y = y - 1
    scores, _ = clustering_metric(y, kmeans_assignments, n_clusters)

    ret = {}
    ret['kmeans'] = scores  # 计算kmeans下的scores
    return y_preds, ret


def calculate_cost_matrix(C, n_clusters):   # 计算代价矩阵，用于匈牙利算法求阶最有聚类标签匹配。
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i  计算真实类别的个数
        for i in range(n_clusters):
            t = C[i, j] # 计算j中被分到i的个数
            cost_matrix[j, i] = s - t   # 聚类类别i被分配到真实类别j中所需要的代价
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """Computes the predicted labels, where label assignments now
        correspond to the actual labels in y_true (as estimated by Munkres)

        Args:
            cluster_assignments: array of labels, outputted by cluster_assignments/kmeans
            y_true:              true labels
            n_clusters:          number of clusters in the dataset

        Returns:
            a tuple containing the accuracy and confusion matrix,
                in that order
    """
    '''
        基于匈牙利算法来获得的y_pre的流程如下:
            计算confusion matrix → 计算矩阵开销 → 将矩阵开销输入Munkres().compute(cost_matrix)，得到匈牙利算法计算出的最有匹配
    '''
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)   # 计算真实标签和聚类标签的混淆矩阵
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)    # 通过匈牙利算法得到最优匹配，格式为：(真实类别，聚类类别)
    # print(indices)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices) # 返回聚类的结果

    if np.min(cluster_assignments) != 0:    # 如果标签是从0开始的，则将标签值进行平移到从0开始
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # recall
    recall = metrics.recall_score(y_true, y_pred, average=average)
    recall = np.round(recall, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix


def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)  # 通过聚类算法得到的聚类标签

    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)

    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}, **classification_metrics), confusion_matrix


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Using either a newly instantiated ClusterClass or a provided cluster_obj, generates
        cluster assignments based on input data.

        Args:
            x: the points with which to perform clustering
            cluster_obj: a pre-fitted instance of a clustering class   已训练好的聚类模型
            ClusterClass: a reference to the sklearn clustering class, necessary
              if instantiating a new clustering class
            n_clusters: number of clusters in the dataset, necessary
                        if instantiating new clustering class
            init_args: any initialization arguments passed to ClusterClass

        Returns:
            返回聚类标签和聚类对象
            a tuple containing the label assignments and the clustering object
    """
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args) # 这里的**用于解包字典中的参数。
        '''关于for-else：只有当for完全运行完，没有被break的时候会进入else；否则被break后将不会进入else'''
        for _ in range(10): # 若连续运行10次都没能正常地训练指定的ClusterClass的话，则会返回else中的内容。
            try:
                cluster_obj.fit(x)
                break  # 若有传入cluster_obj，则会直接调用.fit()进行预测，而后break结束循环；若没有cluster_obj，则会进行循环提醒异常后进入else
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj # 此时

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj

