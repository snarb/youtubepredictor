from Flex_2 import TrainModel, LoadData, GetRegrouppedViews
from SETTINGS import *
import sys

trainDf = LoadData('train')
testDf = LoadData('validation')
trainViewsD, testViewsD = GetRegrouppedViews(trainDf, testDf)

TrainModel(trainViewsD, testViewsD, seqLen=SEQUENCE_LENGTH, minPredDelta=1, maxPredDelta=4, minLen=30, maxLen=None)
# while (True):
#     try:
#         TrainModel(trainViewsD, testViewsD, seqLen = SEQUENCE_LENGTH, minPredDelta = 1, maxPredDelta = 30,  minLen = 30, maxLen = None)
#         break;
#     except:
#         print ("Unexpected error:", sys.exc_info()[0])
