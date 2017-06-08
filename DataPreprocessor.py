import pandas as pd
import numpy as np

def LoadData(name):
    df = pd.read_csv('data/' + name + '.csv',
                     index_col=['date'],
                     dtype={"video_id": object, "views": np.float64},
                     parse_dates=['date'])
    return df


def DfToList(df):
    listOfVideoViews = list(df.groupby('video_id')['views'].apply(pd.DataFrame.as_matrix))
    return listOfVideoViews

def GetRegrouppedViews(trainDf, testDf):
    return DfToList(trainDf), DfToList(testDf)