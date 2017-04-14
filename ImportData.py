import csv
import numpy as np
import json
from enum import Enum
from VideoData import VideoData

SPLIT_PROBS = [0.2, 0.2, 0.6]
SPLIT_NAMES =  ["TEST", "VALID", "TRAIN"]
stat = {"TEST":0, "VALID":0, "TRAIN":0 }

MAX_VIDEOS_PER_CHANNEL = 990000000
MAX_CHANNELS = 30000
PRINT_CH_STEP = 30

with open('data/statisticcollector_20170327.csv', 'rt') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=';')

     currentChannelId = None
     channelVideos = []
     classifiedChannels = 0


     for row in spamreader:
         videoData = VideoData(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10])
         if(not currentChannelId):
             currentChannelId = videoData.channelId

         if(videoData.channelId != currentChannelId):
             subsetName = np.random.choice(SPLIT_NAMES,  p = SPLIT_PROBS)
             stat[subsetName] += len(channelVideos)

             with open("data/" + subsetName, 'a') as targetFile:
                 datawriter = csv.writer(targetFile, delimiter=';')
                 for channelVideo in channelVideos:
                     # json.dump(channelVideo, targetFile)
                     # targetFile.write("\n")
                     datawriter.writerow(channelVideo.ToCSVString())


             currentChannelId = videoData.channelId
             channelVideos.clear()

             if(classifiedChannels % PRINT_CH_STEP == 0):
                 print('Channel: ', classifiedChannels)

             classifiedChannels += 1
             # if(classifiedChannels >= MAX_CHANNELS):
             #     break

         else:
             # if len(channelVideos) < MAX_VIDEOS_PER_CHANNEL:
                channelVideos.append(videoData)


     print(stat)


