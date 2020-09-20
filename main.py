import os
import requests
import json
from pprint import pprint
from requests.auth import AuthBase
from requests.auth import HTTPBasicAuth
from utils.mongo_helper import MongoHelper
from config import logger, TWITTER_API_KEY, TWITTER_API_SECRET
from time import sleep

db = MongoHelper("localhost:27017", db_name="twitter")

consumer_key = TWITTER_API_KEY  # Add your API key here
consumer_secret = TWITTER_API_SECRET  # Add your API secret key here

stream_url = "https://api.twitter.com/labs/1/tweets/stream/sample"


# Gets a bearer token
class BearerTokenAuth(AuthBase):
    def __init__(self, consumer_key, consumer_secret):
        self.bearer_token_url = "https://api.twitter.com/oauth2/token"
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.bearer_token = self.get_bearer_token()

    def get_bearer_token(self):
        response = requests.post(
            self.bearer_token_url,
            auth=(self.consumer_key, self.consumer_secret),
            data={'grant_type': 'client_credentials'},
            headers={"User-Agent": "TwitterDevSampledStreamQuickStartPython"})

        if response.status_code is not 200:
            raise Exception(f"Cannot get a Bearer token (HTTP %d): %s" % (response.status_code, response.text))

        body = response.json()
        return body['access_token']

    def __call__(self, r):
        r.headers['Authorization'] = f"Bearer %s" % self.bearer_token
        return r


def stream_connect(auth):
    response = requests.get(stream_url, auth=auth, headers={"User-Agent": "TwitterDevSampledStreamQuickStartPython"},
                            stream=True)
    for response_line in response.iter_lines():
        if response_line:
            try:
                doc = json.loads(response_line)
                ret = db.insert("rawTweets", doc)
                logger.info(f"Created doc {ret} : inserted tweet")
            except Exception as e:
                logger.error("Error in obtaining tweet", e)
                logger.info("sleeping 1 sec")
                sleep(1)



# Listen to the stream. This reconnection logic will attempt to reconnect as soon as a disconnection is detected.
if __name__ == "__main__":
    bearer_token = BearerTokenAuth(consumer_key, consumer_secret)
    while True:
        logger.info("starting streaming")
        stream_connect(bearer_token)
        logger.error("sleeping before reattempting connection")
        sleep(5)
