from pymongo import MongoClient
import requests
from lxml import etree
#基本配置

client = MongoClient()

meta_db_name = "dpm"
db_name = "jingxuancangpin"
record_db_name = "record"
db = client.get_database(db_name)
meta_db = client.get_database(meta_db_name)
record_db = client.get_database(record_db_name)

meta_col_name = "type_metadata"
meta_col = meta_db.get_collection(meta_col_name)
record_col_name = "{}_type_metadata_crawel".format(db_name)
record_col = record_db.get_collection(record_col_name)

meta = meta_col.find()

for x in meta:
    col_name = x["name"]
    base_url_num = x["base_url_num"]
    base_url = 'http://www.dpm.org.cn/searchs/{}/category_id/{}/p/{}.html'.format(col_name, base_url_num, '{}')
    total_page = x["total_page"]
    col = db.get_collection(col_name)
    # record_col = record_db.get_collection("{}_{}_type_metadata_downloaded".format(db_name, col_name))
    keys = x["keys"]
    print(keys)
    # 遍历网页
    for i in range(total_page + 1):
        url = base_url.format(i + 1)
        if record_col.find({"url": url, "success": True}).count() > 0:
            print('已经爬取过{},第{}页'.format(col_name, i + 1))
            continue

        print('开始爬取{},第{}/{}页'.format(col_name, i + 1, total_page))

        try:
            html = requests.get(url).text
            s = etree.HTML(html)
            trs = s.xpath("//div[@class='building2']//div[@class='table1']/table/tbody/tr")[1:]

            # 提取数据
            items = []
            for tr in trs:
                tds = tr.xpath('./td')
                if len(tds) < 1:
                    print("未获取到数据，{}页,{}".format(i + 1, url))
                    break
                item = {
                    "name": tds[0].xpath('./a/text()')[0].strip(),
                    "href": tds[0].xpath('.//@href')[0],
                    "img_url": tds[0].xpath('.//@src')[0],
                }
                r = len(keys) - 3
                for j in range(r):
                    key = keys[j+3]
                    item[key] = tds[1+j].xpath('./text()')[0].strip()
                items.append(item)
            col.insert_many(items)
            record_col.insert({"url": url, "success": True})
        except Exception as e:
            print(e)
            if not isinstance(e, IndexError):
                record_col.insert({"url": url, "err": str(e), "success": False})
