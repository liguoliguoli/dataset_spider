from pymongo import MongoClient, ASCENDING
import requests
import os

client = MongoClient()
db = client.test_db
collection = db.taibeigugong_opendata_img
download_record_col = db.download_record_col
download_record_col.create_index([("id", ASCENDING)], unique=True)

# print(list(collection.find()[:1]))

taibeigugong_opendata_base_url = 'https://theme.npm.edu.tw/opendata/'
download_img_store_dir = 'G:/pic_data/taibeigugong/full'
if not os.path.exists(download_img_store_dir):
    os.makedirs(download_img_store_dir)

items = collection.find()
for item in items:
    try:
        id = item["id"]
        print(id)
        high_solution_img_download_url = item["high_solution_img_download_url"]
        print(high_solution_img_download_url)
        if download_record_col.find({"id": id, "success": True}).count() > 0:
            print("{}已经下载".format(id))
            continue
        r = requests.get(taibeigugong_opendata_base_url + high_solution_img_download_url)
        print(r)
        download_file = os.path.join(download_img_store_dir, '{}.zip'.format(id))
        with open(download_file, 'wb') as f:
            f.write(r.content)
            download_record_col.insert({"id": id, "success": True})
    except Exception as e:
        print(e)
        try:
            download_record_col.insert({"id": id, "success": False, "errMsg": e})
        except Exception as e:
            print(e)