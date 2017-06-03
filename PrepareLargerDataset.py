import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def FilterData():
    df = pd.read_csv('/home/pkonovalov/prediction/tmp/outfile-001-filtered.csv',
                     sep=',',
                     names=['channel_id', 'video_id', 'datetime', 'youtube_published_at', 'views'],
                     index_col=['datetime'],
                     usecols=['channel_id', 'video_id', 'datetime', 'views'],
                     parse_dates=['datetime']
                     )  # nrows=100

    videoIdsToremove = list(df[df.views == 0]['video_id'])
    #df = df[~df['video_id'].isin(videoIdsToremove)]
    print("o")
    #df2[pd.isnull(df2).any(axis=1)]

    for group in df.groupby(['video_id']):
        videoDf = group[1]
        videoId = group[0]

        minV = videoDf['views'].diff().min()
        if (minV < 0):
            videoIdsToremove.append(videoId)


    pureDf = df[~df['video_id'].isin(videoIdsToremove)]
    pureDf.to_csv('data/big_filtered_set.csv')


def DateSplit():
    df = pd.read_csv('data/big_filtered_set.csv',
                     index_col=['datetime'],
                     dtype = {"channel_id": object, "video_id": object, "views": np.int64},
                     parse_dates=['datetime'],
                     )

    gr = df.groupby(['channel_id', 'video_id', pd.TimeGrouper('D')]).mean()


    gr.to_csv('data/big_filtered_set_by_day.csv')
    print('ok')

def SplitToTestAndTrain():
    df = pd.read_csv('data/big_filtered_by_day.csv',
                     index_col=['datetime'],
                     dtype = {"video_id": object, "views": np.float64},
                     parse_dates=['datetime'],
                     )

    videoIds = df['video_id'].drop_duplicates()

    dfTestIds = videoIds.sample(frac=0.18)
    restDf = videoIds[~videoIds.isin(dfTestIds)]

    dfValidIds = restDf.sample(frac=0.20)

    trainDf = restDf[~restDf.isin(dfValidIds)]

    df[df.video_id.isin(dfTestIds)].to_csv('data/test_ch.csv')
    df[df.video_id.isin(dfValidIds)].to_csv('data/validation_ch.csv')
    df[df.video_id.isin(trainDf)].to_csv('data/train_ch.csv')



df = pd.read_csv('data/big_filtered_set_by_day.csv',
                     index_col=['datetime'],
                     dtype = {"video_id": object, "views": np.float64},
                     parse_dates=['datetime'])

rndChannelId = df.sample()['channel_id'][0]
newDf = df[df.channel_id == rndChannelId]
videosViews = newDf.groupby('video_id')['views'].apply(pd.DataFrame.as_matrix).tolist()
difList = []
for videoViews in videosViews:
    difList.append(np.diff(videoViews))

df = newDf.groupby('video_id')['views'].diff()
chGr = df.groupby(['channel_id', 'video_id'])
res = []
for curCh in chGr:
    curAr = chGr.groupby('video_id')['views'].apply(pd.DataFrame.as_matrix).tolist()
    res.append(curAr)
#SplitToTestAndTrain()
DateSplit()