import json

class VideoData:
    def __init__(self, channelId, channel_subscribers, videoid, date, views, engagements, sentiment, traffic_source, gender, age_groups, country):
        self.channelId = channelId
        self.channel_subscribers = channel_subscribers
        self.videoid = videoid
        self.date = date
        self.views = views
        self.engagements = engagements
        self.sentiment = sentiment
        #self.traffic_source = json.loads(traffic_source)
        #self.gender = json.loads(gender)
        #self.age_groups = json.loads(age_groups)
        #self.country = json.loads(country)


    def ToCSVString(self):
        #return [self.channelId, self.channel_subscribers, self.videoid, self.date, self.views, self.engagements, self.sentiment, json.dumps(self.traffic_source), json.dumps(self.gender), json.dumps(self.age_groups), json.dumps(self.country)]
        return [self.channelId, self.channel_subscribers, self.videoid, self.date, self.views, self.engagements, self.sentiment]
