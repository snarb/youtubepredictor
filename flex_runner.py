
from matplotlib import pyplot as plt
from Flex_2 import TrainModel,TrainModel_2, LoadData, GetRegrouppedViews
from DataPreprocessor import LoadData, GetRegrouppedViews
# from SETTINGS import *
# import sys
from ExploreData import CalcKoefs
from matplotlib.font_manager import FontProperties

trainDf = LoadData('train')



testDf = LoadData('validation')
trainViewsD, testViewsD = GetRegrouppedViews(trainDf, testDf)

ataConf = CalcKoefs(trainViewsD, testViewsD, offset = 30,  predDelta =30)
#
# datas = []
# predDeltas = []
#
# for offset in range(0, 180, 30):
#     for predDelta in range(30, 120, 30):
#         dataConf = CalcKoefs(trainViewsD, testViewsD, offset,  predDelta)
#         datas.append(dataConf)
#         predDeltas.append(predDelta)
#
#     pts = list(zip(*datas))
#     p = plt.plot(predDeltas, pts[0],  linestyle='--', marker='o', label='offset: {0}'.format(offset))
#     col = p[0].get_color()
#     plt.plot(predDeltas, pts[1],  linestyle='--', marker='o', label='offset: {0}'.format(offset), color = col)
#     datas.clear()
#     predDeltas.clear()
#
# plt.legend()
# plt.show()
# print("Done")

#TrainModel(trainViewsD, testViewsD, seqLen=SEQUENCE_LENGTH, minPredDelta=1, maxPredDelta=120, minLen=180, maxLen=None)
TrainModel_2(trainViewsD, testViewsD, minPredDelta=1, maxPredDelta=30, minLen=31, maxLen=None, daysToremove = 0)


# while (True):
#     try:
#         TrainModel(trainViewsD, testViewsD, seqLen = SEQUENCE_LENGTH, minPredDelta = 1, maxPredDelta = 30,  minLen = 30, maxLen = None)
#         break;
#     except:
#         print ("Unexpected error:", sys.exc_info()[0])

