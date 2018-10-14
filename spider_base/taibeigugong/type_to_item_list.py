from pymongo import MongoClient
import requests
from lxml import etree

client = MongoClient()
db = client.test_db

taibeigugong_type_options = db.taibeigugong_type_options
types = taibeigugong_type_options.find()
types = [(x['value'],x['name']) for x in types]
print(types)

taibeigugong_type_to_item_list = db.taibeigugong_type_to_item_list

urls = [(value, name, 'https://theme.npm.edu.tw/opendata/DigitImageSets.aspx'
                      '?Key=^^{}&pageNo='.format(value)) for value, name in types]
print(urls)

try:
    for value, name, url in urls:
        print('开始爬取{}...'.format(value))
        item_list = []
        for i in range(300)[1:]:
            try:
                print('第{}页'.format(i))
                page_url = '{}{}'.format(url, i)
                html = requests.get(page_url).text
                s = etree.HTML(html)
                cur_list = s.xpath("//ul[@class='photoList']/li/a/@href")
                if len(cur_list) < 1:
                    print('爬取{}结束...'.format(value))
                    break
                print(cur_list)
                item_list.extend(cur_list)
            except Exception as e:
                print(e)
        print('{}共爬取到{}个'.format(value, len(item_list)))
        taibeigugong_type_to_item_list.insert({"value": value, "name": name, "list": item_list})
except Exception as e:
    print(e)
