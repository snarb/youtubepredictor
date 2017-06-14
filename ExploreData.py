import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from SETTINGS import *

def CalcKoefs(trainVideos, testVideos, offset, predDelta, seqLen):
    print("Offset:", offset)
    print("Delta:", predDelta)
    #mnkoef = 1.02


    trainVideos = shuffle(trainVideos)
    testVideos = shuffle(testVideos)

    koefs = []
    for video in trainVideos:
        video = np.log(video)
        video = np.clip(video, a_min=STABLE_DELTA, a_max=99999999999)
        #video = video[30:]
        if len(video) > offset + predDelta + seqLen:
            koef =  video[offset + seqLen + predDelta] / video[offset + seqLen]
            koefs.append(koef)
            # if(koef < 5):
            #     koefs.append(koef)

    koefs = np.array(koefs)
    N = 10000
    mean_estimates = []
    for _ in range(N):
        re_sample_idx = np.random.randint(0, len(koefs), koefs.shape)
        mean_estimates.append(np.median(koefs[re_sample_idx]))

    sorted_estimates = np.sort(np.array(mean_estimates))
    conf_median_data_interval = [sorted_estimates[int(0.025 * N)], sorted_estimates[int(0.975 * N)]]
    print("Conf median Int: ", conf_median_data_interval)
    print("mean median conf int", (conf_median_data_interval[0] + conf_median_data_interval[1])/2)



    sorted_data_estimates = np.sort(koefs)
    conf_data_interval = [sorted_data_estimates[int(0.025 * len(sorted_data_estimates))], sorted_data_estimates[int(0.975 * len(sorted_data_estimates))]]
    print("Conf Data Int: ", conf_data_interval)
    print("mean data conf", (conf_data_interval[0] + conf_data_interval[1])/2)
    print("mean ", np.mean(koefs))
    print("median ", np.median(koefs))

    #print("std ", np.std(koefs))
    # print("95 % val",     np.percentile(koefs, 95))
    # print("90 % val",     np.percentile(koefs, 90))
    # print("80 % val", np.percentile(koefs, 80))
    print("---------------------")

    return conf_median_data_interval

    # cnt = (np.array(koefs) > np.mean(koefs) + np.std(koefs)).sum() / len(koefs)
    # print("One std", cnt)
    # cnt = (np.array(koefs) > np.mean(koefs) + 2 * np.std(koefs)).sum() / len(koefs)
    #print("Two std", cnt)
    # n, bins, patches = plt.hist(koefs, 50, facecolor='green')
    #
    #
    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    #
    # plt.show()