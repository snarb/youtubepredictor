import pandas as pd
import numpy as np
import pickle
import sys
# from Constants import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer

def split_to_data_and_lables(dataFrame, time_steps, lable_delta):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    datas = []
    labels = []


    for i in range(len(dataFrame) - time_steps):
        dataEnd = i + time_steps
        lableId = dataEnd + lable_delta - 1
        if lableId < len(dataFrame):
            lable = dataFrame['views'][lableId]
            labels.append(lable)

            curData = dataFrame.iloc[i: dataEnd]
            npAr = curData.as_matrix(['channel_subscribers', 'views', 'engagement_rate', 'sentiment'])
            datas.append(npAr)

    return datas, labels
#
# def split_to_data_and_lables(data, time_steps, labels):


def subsequences(ts, window, lablesShift):
    rowsCount = ts.size - window - lablesShift + 1
    shape = (rowsCount, window)
    strides = (8,)
    data = np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

    lables = ts[window - 1 + lablesShift::1]
    return data, lables

def rolling_window(ts, window, lablesShift):
    DATA_SIZE = ts.itemsize
    inputRowsCount = ts.shape[0]
    columnsCount = ts.shape[1]
    rowsCount = inputRowsCount - window - lablesShift + 1
    shape = (rowsCount, window, columnsCount)
    if(rowsCount < 0):
        r = 4

    try:

        strides = (DATA_SIZE * window,  DATA_SIZE * columnsCount,  DATA_SIZE)
        data = np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

        lables = ts[window - 1 + lablesShift::1, 0] # ,0 - views column index
    except:
        print("Unexpected error:", sys.exc_info()[0])

    return data, lables

def PrepareVideoDfs(fileName, sequenceLength, weeksToPredict):
    # LIMIT = 100
    # CUR_L = 0

    # 1-3-6-12 months.
    df = pd.read_csv('data/' + fileName,
                       sep = ';',
                       names = ['channelId', 'channel_subscribers', 'videoid', 'date', 'views', 'engagements', 'sentiment'],
                       index_col = ['date'],
                       usecols = ['channel_subscribers', 'videoid', 'date', 'views', 'engagements', 'sentiment'],
                       # usecols=['videoid', 'date', 'views'],
                       parse_dates=['date']
                       ) #nrows=100

    # scaler = MinMaxScaler(feature_range=(0.1, 0.8))
    #transformer = FunctionTransformer(np.log)

    # df2 = df.groupby(['videoid', pd.TimeGrouper('M')]).sum()
    #df[df.engagements > 0]['engagement_rate']  = df['engagements'] / df['views']



    #videoDfs = []
    # df.groupby(['videoid', pd.TimeGrouper('W')]).sum()

    # ar1 = df.query('engagements < 0')['videoid'].values
    # ar2 = df[df.views < df.views.shift()]['videoid'].values # views must not become less
    # toDelete = set(np.append(ar1, ar2))
    #
    # df = df.drop(df[df['videoid'].isin(toDelete)].index)

    df['engagement_rate'] = df['engagements'] / df['views']
    #df = df.drop(['engagements'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["engagement_rate"], how="all")

    ar1 = df.query('engagement_rate < 0 or engagement_rate > 75')['videoid'].values
    ar2 = df[(df.views < df.views.shift()) &  (df.videoid == df.videoid.shift())]['videoid'].values
    toDelete = set(np.append(ar1, ar2))
    #df = df.drop(df['videoid'].isin(toDelete).index)


    gr = df.groupby(['videoid', pd.TimeGrouper('W')])
   # gr = gr.filter(lambda x: len(x) >= sequenceLength)
    df = gr.agg({'views': np.mean, 'channel_subscribers': np.mean, 'engagement_rate': np.mean, 'sentiment': np.mean, 'videoid': np.random.choice})
    # df = gr.agg(lambda x: x.iloc[-1])
    # df = df.reset_index()

    #del(gr)
    #df = gr.apply(multi_func({'views': np.sum, 'channel_subscribers': np.mean, 'engagement_rate': np.mean, 'sentiment': np.mean}))
    # df = df.filter(lambda x: len(x) >= sequenceLength)
    # return df

    # g = df.groupby(['videoid', pd.TimeGrouper('W')])
# g.filter(lambda x: len(x) > 1)

    datasL = []
    lablesL = []

    for group in df.groupby(['videoid']):
        # CUR_L += 1
        # if(CUR_L > LIMIT):
        #     return videoDfs
        videoId = group[0]
        videoDf = group[1]
        # videoDf = videoDf.groupby([pd.TimeGrouper('W')]).sum()
        if((videoId not in toDelete) and (len(videoDf) >= sequenceLength + weeksToPredict)):
            minV = videoDf['views'].diff().min()
            if (minV < 0):
                continue

            #datas = videoDf.as_matrix(['views', 'channel_subscribers', 'engagement_rate', 'sentiment'])
            data, lables = split_to_data_and_lables(videoDf, sequenceLength, weeksToPredict)
            #data, lables = rolling_window(datas, sequenceLength, weeksToPredict)


            #videoDf[['views']] = transformer.transform(videoDf[['views']])
            #looks bad! we must scale larger
            # videoDf[['channel_subscribers', 'views', 'engagements', 'sentiment']] = scaler.fit_transform(videoDf[['channel_subscribers', 'views', 'engagements', 'sentiment']])
            # datas = data.as_matrix(['channel_subscribers', 'views', 'engagement_rate', 'sentiment'])
            # del data
            # lable = lables.as_matrix(['views'])
            # del lables

            datasL.append(data)
            lablesL.append(lables)


    return datasL, lablesL


def DumpTarget(fileName, sequenceLength, weeksToPredict):
    all_training_data, all_lables = PrepareVideoDfs(fileName, sequenceLength, weeksToPredict)

    # all_training_data = np.concatenate(all_training_data,  axis=0)
    # all_lables = np.concatenate(all_lables,  axis=0)
    np.save('data/' + fileName + "_s:" + str(sequenceLength) + "_p:" + str(weeksToPredict) + "_training_data", all_training_data)
    np.save('data/' + fileName + "_s:" + str(sequenceLength) + "_p:" + str(weeksToPredict) + "_lables", all_lables)

    # DumpTargetInternal(videoDfs, fileName, sequenceLength, weeksToPredict)

def DumpTargetInternal(videoDfs, fileName, sequenceLength, weeksToPredict):
    all_training_data = []
    all_lables = []

    for videoDf in videoDfs:
        training_data, lables = split_to_data_and_lables(videoDf, sequenceLength, weeksToPredict)

        if(len(training_data) > 0):
            training_data = np.stack(training_data)
            lables = np.stack(lables)

            all_training_data.append(training_data)
            all_lables.append(lables)

    all_training_data = np.concatenate(all_training_data,  axis=0)
    all_lables = np.concatenate(all_lables,  axis=0)
    np.save('data/' + fileName + "_s:" + str(sequenceLength) + "_p:" + str(weeksToPredict) + "_training_data", all_training_data)
    np.save('data/' + fileName + "_s:" + str(sequenceLength) + "_p:" + str(weeksToPredict) + "_lables", all_lables)



DumpTarget("TEST", sequenceLength = 8, weeksToPredict=4)
DumpTarget("TRAIN", sequenceLength = 8, weeksToPredict=4)

print("Done 1")
DumpTarget("TEST", sequenceLength = 8, weeksToPredict=1)
DumpTarget("TRAIN", sequenceLength = 8, weeksToPredict=1)

DumpTarget("TEST", sequenceLength = 12, weeksToPredict=4)
DumpTarget("TRAIN", sequenceLength = 12, weeksToPredict=4)
#
# DumpTarget("TEST", sequenceLength = 16, weeksToPredict=4)
# DumpTarget("TRAIN", sequenceLength = 16, weeksToPredict=4)
#
DumpTarget("TEST", sequenceLength = 16, weeksToPredict=12)
DumpTarget("TRAIN", sequenceLength = 16, weeksToPredict=12)

# DumpTarget("TEST", sequenceLength = 16, weeksToPredict=12)
# DumpTarget("TRAIN", sequenceLength = 16, weeksToPredict=12)

print("Done")

