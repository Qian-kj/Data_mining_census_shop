import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

#2.1
dataset = pd.read_csv('C:/Users/qiankj/Desktop/KCL/7 Data mining/Coursework/data/wholesale_customers.csv',header = 0)
wsc_X = dataset.copy()

#drop the attributes Channel & Region
wsc_X.drop(['Channel','Region'],axis=1,inplace=True)
print(wsc_X.head(5))

#calculate the mean of each attribute
wsc_mean = wsc_X.mean()
print("The mean for each attribute:\n", wsc_mean)

#calculate the minimun and maximum of each attribute
wsc_min = wsc_X.min()
wsc_max = wsc_X.max()
print("The maximum for each attribute:\n", wsc_max)
print("The minimum for each attribute:\n", wsc_min)

#2.2
PLOTS_DIR = 'C:/Users/qiankj/Desktop/KCL/7 Data mining/Coursework/plots/'
n_clusters = 3

#define markers for up to 10 clusters
CLUSTER_MARKERS = [ 'bo', 'rv', 'c^', 'm<', 'y>', 'ks', 'bp', 'r*', 'cD', 'mP' ]

#number of instances
M = len(wsc_X)

#list of attributes' names
attr = list(wsc_X)

km = cluster.KMeans(n_clusters=n_clusters, random_state=10).fit(wsc_X)
# cluster labels
cluster_labels = km.labels_
ls = []
m = 0
for i in range(6):
    ls.append(attr[i])
    for j in range(6):
        if attr[i] != attr[j] and attr[j] not in ls:
            X = wsc_X[[attr[i], attr[j]]]

            # Transform dataframe to list
            X_list = np.array(X).tolist()
            # scatter plot
            plt.figure(figsize=(6, 5))
            for n in range(M):
                plt.scatter(X_list[n][0], X_list[n][1],
                            marker=CLUSTER_MARKERS[cluster_labels[n]][1],  # different point shape for each cluster
                            c=CLUSTER_MARKERS[cluster_labels[n]][0],  # different colours for each cluster
                            alpha=0.6)

            plt.xlabel(attr[i])
            plt.ylabel(attr[j])
            m += 1
            plt.title('Fig.2.{} The annual expenses of {} vs {}'.format(m,attr[i], attr[j]))
            plt.savefig(PLOTS_DIR + '{} vs {}.png'.format(attr[i], attr[j]))
            plt.show()
            plt.close()

#2.3
#define a function to calculate the between-cluster score
def ExtraIntraDistance(km, K):
    between = np.zeros((K))
    # loop through all clusters
    for i in range(K):
        between[i] = 0.0
        # loop through remaining clusters
        for l in range(i + 1, K):
            between[i] += (np.square(km.cluster_centers_[i][0] - km.cluster_centers_[l][0]) +
                           np.square(km.cluster_centers_[i][1] - km.cluster_centers_[l][1]))
    BC = np.sum(between)
    return BC

K_list = [3, 5, 10]
BC_ls = []
WC_ls = []
ratio_ls = []
# loop through the set of k values
for K in K_list:
    # create k-means model
    km = cluster.KMeans(n_clusters=K, random_state=10).fit(wsc_X)
    # cluster labels
    cluster_labels = km.labels_
    # calculate the within-cluster score
    WC = km.inertia_
    WC_ls.append(WC)
    # calculate the Between-cluster score
    BC = ExtraIntraDistance(km, K)
    BC_ls.append(BC)
    # calculate the ratio of BC/WC
    ratio = BC/WC
    ratio_ls.append(ratio)

# Put BC, WC, BC/WC into a dataframe
tab = [BC_ls, WC_ls, ratio_ls]
tab_df = pd.DataFrame(np.array(tab), index = ['BC','WC','BC/WC'], columns=[3,5,10])
print(tab_df)

