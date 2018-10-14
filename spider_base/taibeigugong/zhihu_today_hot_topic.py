import requests
from pymongo import MongoClient,ASCENDING
import urllib
client = MongoClient()
db = client.test_db
collection = db.test_zhihu_today_hot_topic
collection.create_index([('id', ASCENDING)], unique=True)

nb_urls = 500
headers = {
    'cookie': '_zap=6849c5fe-51ad-4de1-82ce-2913625df5d9; '
             '_xsrf=Xa1YJVThgR7YaMmnToJmWrLZkZq3mm9z; d_c0'
             '="ADBlbOI9JQ6PTgGt4mvyxkwklGqzdaVCQBk=|15357'
             '97419"; capsion_ticket="2|1:0|10:1535797427|1'
             '4:capsion_ticket|44:MjdkZTQ3NTJiNWM4NDllMThkZ'
             'jMyNzZjMzMyYTgwMTA=|53b5ac5ad5de2ce4041085782'
             'fdbccc4730493f7e5f57e5b554a66e4f62835ba"; z_c0'
             '="2|1:0|10:1535797469|4:z_c0|92:Mi4xNUxxakF3QU'
             'FBQUFBTUdWczRqMGxEaVlBQUFCZ0FsVk4zYlozWEFBbWhp'
             'bVFNZ0xDLUNsX2hxVXZkUXdJZTVpVDVB|91c019de073e0'
             '2274ed855531084a620a97cf308f79b9fa5c51e789a46'
             '0a92d0"; q_c1=7b90a2c10f614122993c54d47ded4f13|'
             '1535797469000|1535797469000; __utmc=51854390; _'
             '_utmv=51854390.100--|2=registration_date=201610'
             '31=1^3=entry_date=20161031=1; tgw_l7_route=29b9'
             '5235203ffc15742abb84032d7e75; __utma=51854390.2'
             '004205633.1535856042.1535860289.1535884238.3; _'
             '_utmb=51854390.0.10.1535884238; __utmz=51854390'
             '.1535884238.3.3.utmcsr=zhihu.com|utmccn=(referra'
             'l)|utmcmd=referral|utmcct=/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                 'AppleWebKit/537.36 (KHTML, like Gecko) Chr'
                 'ome/68.0.3440.106 Safari/537.36'
}
for i in range(nb_urls):
    url = 'https://www.zhihu.com/node/' \
          'ExploreAnswerListV2?params={}'.format(urllib.parse.quote('{"offset":%s,"type":"day"}'%(i*5)))
    try:
        res = requests.get(url, headers=headers)
        data = res.text
        print('正在爬取第{}个url'.format(i))
        # print(data)
        collection.insert_one({"id": i, "content": data})
    except Exception as e:
        print(e)