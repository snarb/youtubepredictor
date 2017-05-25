import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from math import floor
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans

UNLIMITED = 10e15


# Load data from the CSV files into a pandas DataFrame
def LoadData():
    df = pd.read_csv('data/big_filtered_set_by_day.csv',
                     index_col=['datetime'],
                     dtype={"video_id": object, "views": np.float64},
                     parse_dates=['datetime'])
    return df


# Convert DataFame to list of videos views arrays
def DfToViews(df):
    return df.groupby('video_id')['views'].apply(pd.DataFrame.as_matrix).tolist()


# Convert DataFame total views to daily views
def ToDailyViews(videosViews):
    difList = []
    for videoViews in videosViews:
        difList.append(np.diff(videoViews))

    return difList


def GetChannelVideos(df, channelId, daily=False):
    chDf = df[df.channel_id == channelId]
    videosViews = DfToViews(chDf)

    if (daily):
        videosViews = ToDailyViews(videosViews)

    return videosViews


# Exctact seqLength days of views history with
# removed sequences with more then maxLimit days and less then minLimit.
def ExtractNormilizedSequences(listOfVideoViews, seqLen, minLen, maxLen):
    if (not minLen):
        minLen = 0

    if (not maxLen):
        maxLen = 10e+10

    datas = []
    lables = []
    for s in listOfVideoViews:
        if (len(s) > minLen and len(s) < maxLen):
            data = np.array(s[:seqLen])
            maxData = np.max(data)  # data[seqLen - 1]
            if (maxData > 0):
                data = data / maxData
                datas.append(data)

    return np.array(datas)


# Calculating K-NN clusters and printing plots with results.
# df - target DataFrame. seqLength, minLimit, maxLimit, clustersCount -the same as in ExtractNormilizedSequences.
def PrintClusters(df, seqLength, minLimit, maxLimit, clustersCount):
    EXAMPLES_TO_SHOW = 3
    train_inputs = ExtractNormilizedSequences(df, seqLength, minLimit, maxLimit)
    kmeans = KMeans(n_clusters=clustersCount).fit(train_inputs)  # , random_state=0
    for i in range(clustersCount):
        targets = np.where(kmeans.labels_ == i)[0]
        print('Cluster N: ', i)
        targetAr = train_inputs[targets]
        menVals = np.mean(targetAr, axis=0)
        stdVals = np.std(targetAr, axis=0)
        stdMean = stdVals.mean()

        goodList = []
        for y in range(len(targetAr)):
            curStd = stdVals[y]

            if (curStd < stdMean * 2):
                print(targetAr[y])
                goodList.append(targetAr[y])
                if (len(goodList) >= EXAMPLES_TO_SHOW):
                    break

        plt.plot(goodList)
        plt.plot(menVals)
        plt.show()
        print("Cluster popularity: {0:.2f}% \n".format(len(targets) * 100.0 / len(kmeans.labels_)))


df = LoadData()
totalViews = DfToViews(df) # Total views
PrintClusters(totalViews, seqLength = 360, minLimit = 360, maxLimit = UNLIMITED, clustersCount = 6)
PrintClusters(totalViews, seqLength = 360, minLimit = 360, maxLimit = UNLIMITED, clustersCount = 6)