from pymongo import MongoClient
import requests
from lxml import etree
import os
import time


def download_minghuaji_img():
    db_name = "dpm"
    col_name = "minghuaji"
    record_db_name = "record"
    record_col_name = "dpm_minghuaji"
    data_base_dir = "G:\\pic_data\\dpm\\origin\\minghuaji"

    client = MongoClient()
    db = client.get_database(db_name)
    col = db.get_collection(col_name)
    record_db = client.get_database(record_db_name)
    record_col = record_db.get_collection(record_col_name)

    if not os.path.exists(data_base_dir):
        os.makedirs(data_base_dir)

    metas = list(col.find())
    # print(metas)
    img_base_url = "http://minghuaji.dpm.org.cn/image-bundle"

    for x in metas:
        name = x["title"]
        url = x["onclick"].split("'")[1]
        id = x["id"].split("-")[1]
        print("{:<30} {:<} {:<}".format(name, id, url))
        col_key = name
        if record_col.find({col_key:True}).count() > 0:
            print("{}已经下载".format(col_key))
            continue

        for scale in range(20):
            scale_key = "{}/{}".format(name, scale)
            if record_col.find({scale_key:True}).count() > 0:
                print("{}已经下载".format(scale_key))
                continue

            s_not_end = False
            for i in range(1000):
                i_key = "{}/{}/{}".format(name, scale, i)
                if record_col.find({i_key: True}).count() > 0:
                    print("{}已经下载".format(i_key))
                    continue

                i_not_end = False
                for j in range(1000):
                    try:
                        img_url = "{}/{}/{}/{}_{}.jpg".format(img_base_url, id, scale, i, j)
                        if record_col.find({"url": img_url, "success": True}).count() > 0:
                            print("{}已下载".format(img_url))
                            i_not_end = True
                            continue
                        r = requests.get(img_url)

                        if r.status_code == 200 and r.headers["Content-Type"] in ["image/jpeg", ]:
                            i_not_end = True
                            s_dir = "{}/{}/{}".format(data_base_dir, name, scale)
                            if not os.path.exists(s_dir):
                                os.makedirs(s_dir)
                            dst_file = "{}/{}_{}.jpg".format(s_dir, i, j)
                            with open(dst_file, 'wb') as f:
                                f.write(r.content)
                            record_col.insert_one({
                                "url": r.url,
                                "success": True
                            })
                            print("下载{}".format(dst_file))
                        else:
                            print(scale, i, j, img_url)
                            # print(r.url, r.status_code)
                            record_col.insert_one({
                                "url": r.url,
                                "success": False,
                                "status_code": r.status_code
                            })
                            break
                    except Exception as e:
                        print(e)
                        record_col.insert_one({
                            "url": img_url,
                            "success": False,
                            "err": str(e)
                        })

                if i_not_end:
                    s_not_end = True
                    record_col.insert_one({i_key: True})
                    continue
                else:
                    break

            if s_not_end:
                record_col.insert_one({scale_key: True})
                continue
            else:
                break

        record_col.insert_one({col_key: True})


if __name__ == '__main__':
    download_minghuaji_img()
