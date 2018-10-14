from pymongo import MongoClient
import requests
from lxml import etree


db_name = "dpm"
col_name = "minghuaji"

client = MongoClient()
db = client.get_database(db_name)
col = db.get_collection(col_name)

total_page = 23
base_url = "http://minghuaji.dpm.org.cn/?dynasty=0&page={}"

for i in range(total_page):
    url = base_url.format(i+1)
    print("准备爬取第{}页".format(i+1))

    try:
        html = requests.get(url).text
        s = etree.HTML(html)

        items = s.xpath("//div[@id='anchor']//div[@class='wrapper style2 spotlights']")

        for item in items:
            x = {
                "onclick": item.xpath("./@onclick")[0],
                "id": item.xpath("./section/@id")[0],
                "title": item.xpath("./section//h2/text()")[0].strip(),
                "desc": item.xpath("./section//p/text()")[0].strip()
            }
            print(x)
            col.insert_one(x)
    except Exception as e:
        print(e)
