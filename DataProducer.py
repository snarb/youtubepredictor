import numpy as np
from random import choice
from SETTINGS import *
from itertools import repeat

class DataProducer:

    def __init__(self, videos, seqLen, deltasToExtractCount, minPredDelta, maxPredDelta,  minLen, maxLen):
        self.videos = videos
        self.minPredDelta = minPredDelta
        self.maxPredDelta = maxPredDelta
        self.minLen = minLen
        self.seqLen = seqLen
        self.deltasToExtractCount = deltasToExtractCount

        if (not self.minLen):
            self.minLen = 0

        self.maxLen = maxLen

        if (not self.maxLen):
            self.maxLen = 10e+10

        self.currentVideoIndex = 0
        self.currentBatchIndex = -1
        self.batchesStack = []

    def rolling_window(a, window, lablesShift):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)

        data = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        lables = a[window - 1 + lablesShift::1]
        return data[:-lablesShift], lables

    # for "pairs" of any length
    def chunkwise(t, size):
        it = iter(t)
        return zip(*[it] * size)

    def ExtractSequenes(self, video):
        datas = []
        lables = []
        deltasToExtract = []

        for deltaToExtract in range(self.deltasToExtractCount):
            predDelta = choice(range(self.minPredDelta, self.maxPredDelta))
            data, lable = DataProducer.rolling_window(video, window=self.seqLen, lablesShift=predDelta)
            datas.extend(data)
            lables.extend(lable)
            deltasToExtract.extend(repeat(predDelta, len(lable)))

        datas_res = np.clip(np.log(datas), a_min=STABLE_DELTA, a_max=99999999999)
        lables_res = np.clip(np.log(lables), a_min=STABLE_DELTA, a_max=99999999999)

        return zip(datas_res, lables_res, deltasToExtract)

    def FillStack(self):
        while True:
            if (self.currentVideoIndex > len(self.videos)):
                self.currentVideoIndex = 0
                print("End of dataset reached.")

            currentVideo = self.videos[self.currentVideoIndex][DAYS_TO_REMOVE:]
            self.currentVideoIndex += 1

            if (len(currentVideo) > self.minLen and len(currentVideo) < self.maxLen):
                trainingData = list(self.ExtractSequenes(currentVideo))
                batches = list(DataProducer.chunkwise(trainingData, size=BATCH_SIZE))
                if(len(batches) > 0):
                    self.batchesStack.extend(batches)
                    break


    def GetNextBatch(self):
        if(len(self.batchesStack) == 0):
            self.FillStack()

        batch = self.batchesStack.pop()
        return batch