#coding:utf-8
from pymongo import MongoClient,ASCENDING
import requests
import json
from lxml import etree

client = MongoClient()
meda_db_name = "dpm"
meta_col_name = "metadata"
meta_db = client.get_database(meda_db_name)
meta_col = meta_db.get_collection(meta_col_name)
db_name = "cangpinzongmu"
db = client.get_database(db_name)

url_tmpt = "http://img.dpm.org.cn/Public/static/CCP/json/{}/{}_{}.js"

mbrs = meta_col.find({"key":"cangpinzongmu_count"})[0]["value"]["mbrs"]
col_cn_names = [x.split("_")[0] for x in mbrs.keys() if x.endswith("rows")]
col_map = {}
for col_cn_name in col_cn_names:
    col_en_name = mbrs[col_cn_name]
    col_total_count = mbrs[col_cn_name+"_totalrows"]
    col_map[col_en_name] = col_total_count

print(mbrs)

for col_name in col_map.keys():
    col = db.get_collection(col_name)
    col.create_index([("rnum",ASCENDING)],unique=True)
    col_total_count = col_map[col_name]
    if col.find().count() == col_total_count:
        print("{}已经爬取完成{}个".format(col_name,col_total_count))
        continue
    # # 最终统计
    # # else:
    # #     print("{}已爬取{}/{}个".format(col_name,col.find().count(),col_total_count))
    # #     continue
    if col_name in ["gujiwenxian"]:
        continue
    print("准备爬取{},共{}个,已爬取{}个".format(col_name, col_total_count, col.find().count()))

    i = 0
    count = col.find().count()
    while count < col_total_count:
        i += 1
        try:
            url = url_tmpt.format(col_name, col_name, i)
            print("准备获取第{}个js文件:{}".format(i, url))
            js = requests.get(url).text
            js = js.split("=")[1][:-1]
            r = json.loads(js)
            keys = r.keys()
            for key in r.keys():
                if key.startswith("obj"):
                    values = r[key]
            # print(values)
            col.insert_many(values)
            count += len(values)
            print("{}已爬取{}/{}".format(col_name,count, col_total_count))
        except Exception as e:
            print(e)

print("最终统计结果---------------------")
all_count = 0
for col_name in col_map.keys():
    col = db.get_collection(col_name)
    col_total_count = col_map[col_name]
    if col.find().count() == col_total_count:
        print("{}已经爬取完成{}个".format(col_name,col_total_count))
    else:
        print("{}已爬取{}/{}个".format(col_name,col.find().count(),col_total_count))
    all_count += col.find().count()

print("共计{}个".format(all_count))