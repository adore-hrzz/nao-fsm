import time
import math
import numpy as np
import sys
import matplotlib
import Pycluster
import matplotlib.pyplot as plt
from scipy import stats
from ghmm import *

def matrices(state_num, output_num):
    sigma = IntegerRange(0, output_num)
# matrica A #####################################################################
    A = np.matrix(np.identity(state_num)) / float(2)

    for i in range(state_num - 1):
        A[i, i+1] = float(1)/2

    A[state_num-1,state_num-1] = 1

    #print A
# matrica B #####################################################################
    B = np.ones((state_num, output_num)) / float(output_num)
    #print B
# pi ############################################################################
    A = A.tolist()
    B = B.tolist()
    pi = np.ones((state_num))/float(state_num)
    pi = pi.tolist()
    #pi[0] = 1
    #print pi

    return sigma, A, B, pi


def sort_labels(labels):
    existing_labels = [labels[0]]
    for label in labels:
        try:
            existing_labels.index(label)
        except ValueError:
            existing_labels.append(label)
    sorted_labels = sorted(existing_labels)
    ordered_labels = np.asarray([sorted_labels[existing_labels.index(element)] for element in labels])

    return ordered_labels

def cluster(fname, nclust):
    fh = open(fname, 'r')
    lines = fh.readlines()
    fh.close()

    clusters = int(nclust)

    points = []
    points_r = []
    dates = []
    volumes = []
    close_prices = []

    for i in range(len(lines)):
            if i <= 1:
                    continue
            line_c = lines[i-1].strip().split(',')
            close_price = float(line_c[0])
            volume = float(line_c[1])

            points_r.append((close_price, volume))
            volumes.append(volume)
            close_prices.append(close_price)
            #dates.append(line_c[0])

    volume_z= np.array(volumes)
    #volume_z = stats.zscore(a)
    close_price_z = np.array(close_prices)
    #close_price_z = stats.zscore(a)

    points = zip(close_price_z, volume_z)

    init_data = []
    k = len(points) / (nclust)

    for i in range(nclust - 1):
        for j in range(k):
            init_data.append(i)

    while(len(points) != len(init_data)):
        init_data.append(nclust-1)
    #print(clusters)

    labels, error, nfound = Pycluster.kcluster(points, clusters, None, None, 0, 1, 'a', 'e', init_data)
    labels_sorted = sort_labels(labels)
    #print('Labels: ')
    print labels_sorted
    return labels_sorted


if __name__ == '__main__':
    state_num = input('Broj stanja: ')
    output_num = input('Broj izlaza: ')

    dataset_num = 101 # broj fileova data seta
    clust_data = []
    clust_set = []
    gesture_seq = []
    gesture_treshold = 0

    for i in range(dataset_num):
        clust_data.append(cluster("./geste/Drink/gest" + str(i) + ".txt", output_num)) # file-ovi za trening

        #clust_data.append(cluster("gest" + str(i) + ".txt", output_num))

    for i in range(dataset_num):
        clust_set = np.concatenate((clust_set, clust_data[i]), 0)

    [sigma, A, B, pi] = matrices(state_num, output_num)

    m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)
    #print m
# train sequence #################################################################
    train_seq = EmissionSequence(sigma, clust_set.tolist())
    #print train_seq

    m.baumWelch(train_seq)

    # m.write('./geste/Drink/m_file.xml') # izrada m filea
    #print m

    for i in range(dataset_num):
        gesture_seq.append(EmissionSequence(sigma, clust_data[i].tolist()))

    #for i in range(dataset_num):
    #    print m.loglikelihood(gesture_seq[i])
    #    gesture_treshold = gesture_treshold + m.loglikelihood(gesture_seq[i])
    #gesture_treshold = gesture_treshold / float(dataset_num)

    for i in range(dataset_num):
        #print('###########################################')
        #print m.viterbi(gesture_seq[i])
        gesture_treshold = gesture_treshold + m.viterbi(gesture_seq[i])[1]
    gesture_treshold = gesture_treshold / float(dataset_num)
    #print gesture_treshold

    # f = open('./geste/Drink/gesture_file.txt', 'w') # podaci za gestu, treshold i broj stanja
    # f.writelines(str(state_num))
    # f.write('\n')
    # f.write(str(output_num))
    # f.write('\n')
    # f.write(str(gesture_treshold))
    # f.close()
