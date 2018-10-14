from pymongo import MongoClient
import os
import requests

db_name = "jingxuancangpin"
meta_col_name = "type_metadata"
record_db_name = "record"
record_col_name = "dpm_type_img"
base_url = 'http://www.dpm.org.cn'
base_dir = r"G:\pic_data\dpm\origin\type23"


client = MongoClient()
db = client.get_database(db_name)
meta_col = db.get_collection(meta_col_name)
record_db = client.get_database(record_db_name)
record_col = record_db.get_collection(record_col_name)

# meta = meta_col.find()
# col_names = []
# for x in meta:
#     col_names.append(x["name"])
# print(col_names)

for col_name in db.collection_names():
    print('开始下载{}-------------------------------------------'.format(col_name))
    col = db.get_collection(col_name)

    col_dir = os.path.join(base_dir, col_name)
    if not os.path.exists(col_dir):
        os.mkdir(col_dir)

    for x in col.find():
        try:
            file_name = os.path.join(col_dir, x["img_url"].split('.')[0].split('/')[-1] + '.jpg')
            url = base_url + x["img_url"]
            if os.path.exists(file_name):
                print("{}已经下载".format(file_name))
                continue

            print("开始下载", url)
            r = requests.get(url)
            with open(file_name, 'wb') as f:
                f.write(r.content)
                print('{}下载完成'.format(file_name))
        except Exception as e:
            print(e)
            record_col.insert_one({"url": url, "success": False, "err": str(e)})