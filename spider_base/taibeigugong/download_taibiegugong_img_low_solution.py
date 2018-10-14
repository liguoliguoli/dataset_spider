from pymongo import MongoClient, ASCENDING
import requests
import os

client = MongoClient()
db = client.test_db
collection = db.taibeigugong_opendata_img
download_record_col = db.download_record_col_low_solution
download_record_col.create_index([("id", ASCENDING)], unique=True)

# print(list(collection.find()[:1]))

download_img_store_dir = 'G:/pic_data/taibeigugong/low_solution'
if not os.path.exists(download_img_store_dir):
    os.makedirs(download_img_store_dir)

items = collection.find()
for item in items:
    try:
        id = item["id"]
        print(id)
        low_solution_img_download_url = item["low_solution_img_download_url"]
        print(low_solution_img_download_url)
        if download_record_col.find({"id": id, "success": True}).count() > 0:
            print("{}已经下载".format(id))
            continue
        r = requests.get(low_solution_img_download_url)
        print(r)
        download_file = os.path.join(download_img_store_dir, '{}.jpg'.format(id))
        with open(download_file, 'wb') as f:
            f.write(r.content)
            download_record_col.insert({"id": id, "success": True})
    except Exception as e:
        print(e)
        try:
            download_record_col.insert({"id": id, "success": False, "errMsg": e})
        except Exception as e:
            print(e)