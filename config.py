import os
import logging


APP_NAME = 'TwitterTopicModel'

# MongoDB
MONGODB_HOST_PORT = os.getenv("MONGODB_HOST_PORT", "localhost:27017")
MONGO_DB_NAME = os.getenv("DB_NAME", "twitter")

MONGODB_TWEETS_COLLECTION = "processedTweetsEN"
MONGODB_RAW_TWEETS_COLLECTION = "rawTweets"

logging.basicConfig(format='[%(asctime)s.%(msecs)03d %(levelname)s %(name)s:%(filename)s:%(lineno)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level="INFO")
logger = logging.getLogger(APP_NAME)

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")