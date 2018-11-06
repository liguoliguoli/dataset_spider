from pymongo import MongoClient
import requests
import os

client = MongoClient()
db = client.get_database("dpm_all")
col = db.get_collection("img_all")

headers = {

}

data_dir = r"I:\img\dpm\origin\type_all"
base_url = "http://www.dpm.org.cn/"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

for x in col.find():
    url = base_url + x["img_url"]
    id = x["img_id"]
    f_name = "{}\\{}.jpg".format(data_dir, id)

    if os.path.exists(f_name):
        print(url, "已经下载")
        continue
    print("下载",  url)

    try:
        res = requests.get(url, headers=headers)
        if not res.headers["Content-Type"].startswith("image"):
            print("非图片类型", res.headers["Content-Type"])
            continue
        data = res.content
        with open(f_name, 'wb') as f_img:
            f_img.write(data)
    except Exception as e:
        print(e)