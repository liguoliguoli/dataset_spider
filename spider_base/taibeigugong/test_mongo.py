from pymongo import MongoClient,ASCENDING
client = MongoClient()
db = client.test_db
collections = db.test_set

# collections.delete_many({'name': '立国'})
# collections.create_index([('name', ASCENDING)], unique=True)
# collections.insert({'name': '立国'})
collections.update_many({'age': 10}, {'$set': {'height': 100}}, upsert=True)