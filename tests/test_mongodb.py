import pytest
from utils.mongo_helper import MongoHelper

def test_mongodb():
    db = MongoHelper("localhost:27017", db_name="test")
    test_data = {"key": "value"}
    ret = db.insert("tests", test_data)
    print(ret)
