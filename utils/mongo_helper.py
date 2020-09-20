import os
import time

from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateMany
from pymongo.errors import BulkWriteError
from config import (
    MONGODB_HOST_PORT,
    MONGO_DB_NAME,
    logger
)

class MongoHelper():
    def __init__(self, host, user="", password="", db_name="", kwargs=""):
        self.db = self._get_mongo(host, user, password, db_name, kwargs)

    @staticmethod
    def _get_mongo(host, user=None, password=None, db_name=None, kwargs_str=""):
        '''
        host is hostname:port or a comma separated list of host:port
        kwargs_str is in the form of `ssl=true&some_other=false`
        '''
        max_pool_size = 100
        param = {"maxPoolSize": max_pool_size}
        if user and password:
            logger.debug("Use login credentials")
            param.update({
                "socketTimeoutMS": 6000, # almost infinite timeout
                "connectTimeoutMS": 6000
            })
        if os.getpid() != 0:
            logger.debug("This is a forked process")
            param.update({"connect": False})  # No connect if forked process
        host_uri = f"mongodb://{host}/?{kwargs_str}"
        client = MongoClient(host_uri, **param)

        logger.info(client)
        logger.info("MongoDB successfully connected")
        db = client[db_name]
        return db

    def get_all(self, collection):
        return list(self.db[collection].find({}))

    def select(self, collection, filter, first=False, sort=None, limit=None):
        d = {}
        for k, v in filter.items():
            if isinstance(v, list):
                if len(v) > 0:
                    d.update({k: {"$in": v}})
            else:
                d.update({k: v})
        if first:
            result = self.db[collection].find_one(d)
        else:
            if limit:
                kwargs = {"limit": limit}
            else:
                kwargs = {}
            result = self.db[collection].find(d, **kwargs)  # convert Cursor to list
            if sort:
                result = result.sort(sort, ASCENDING)
            result = list(result)
        return result

    def insert(self, collection, post):
        if isinstance(post, list):
            result = self.db[collection].insert_many(post)
            return result.inserted_ids
        else:
            result = self.db[collection].insert_one(post)
            return result.inserted_id

    def upsert(self, collection, filter, post, unset=False, upsert=True):
        if unset:
            operator = "$unset"
        else:
            operator = "$set"
        result = self.db[collection].update_one(filter, {operator: post}, upsert=upsert)
        return result.raw_result

    def upsert_all(self, collection, filter, post, unset=False, upsert=True):
        if unset:
            operator = "$unset"
        else:
            operator = "$set"
        result = self.db[collection].update_many(filter, {operator: post}, upsert=upsert)
        return result.upserted_id

    def bulk_update(self, collection, filters, posts, unset=False, upsert=True):
        if unset:
            operator = "$unset"
        else:
            operator = "$set"

        filter_posts = list(zip(filters, posts))
        requests = [UpdateMany(x[0], {operator: x[1]}, upsert=upsert) for x in filter_posts]
        try:
            self.db[collection].bulk_write(requests)
        except BulkWriteError as e:
            logger.exception(e.details)

    def delete(self, collection, filter):
        result = self.db[collection].delete_one(filter)
        return result.deleted_count

    def delete_all(self, collection, filter):
        result = self.db[collection].delete_many(filter)
        return result.deleted_count

    def reset(self, collection):
        result = self.db[collection].delete_many({})
        return result.deleted_count

    def count(self, collection, filters={}, groupby=[]):
        group = {
            "_id": {key: ('$%s' % key) for key in groupby} or {'None': '$None'},
            "count": {"$sum": 1}
        }
        result = self.db[collection].aggregate([
            {"$match": filters},
            {"$group": group}
        ])
        return list(result)

    def create_index(self, collection, criteria):
        self.db[collection].create_index(criteria)



# Create the Mongo Helper singleton the first time it is used.
