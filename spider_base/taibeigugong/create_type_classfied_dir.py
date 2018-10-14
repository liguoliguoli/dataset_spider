import os
import shutil
from pymongo import MongoClient

client = MongoClient()
db = client.test_db
col = db.taibeigugong_type_to_item_list
type_to_item_list = [x for x in col.find()]

type_classfied_base_dir = "G:\\pic_data\\taibeigugong\\classfied\\type\\low_solution"
source_dir = "G:\\pic_data\\taibeigugong\\low_solution"

for subtype in type_to_item_list:
    sub_dir = os.path.join(type_classfied_base_dir, "low_solution\\{}".format(subtype['value']))
    print(sub_dir)
    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)
    os.makedirs(sub_dir)
    for id in subtype['list']:
        src_jpg = os.path.join(source_dir, "{}.jpg".format(id))
        print(src_jpg)
        dst_jpg = os.path.join(sub_dir, "{}.jpg".format(id))
        print(dst_jpg)
        try:
            shutil.copy2(src_jpg, dst_jpg)
        except Exception as e:
            print(e)