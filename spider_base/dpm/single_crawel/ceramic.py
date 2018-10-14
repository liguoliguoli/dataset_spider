from pymongo import MongoClient
import requests
from lxml import etree

client = MongoClient()
db = client.dpm
col = db.ceramic
col_record = db.ceramic_downloaded

ceramic_base_url = 'http://www.dpm.org.cn/searchs/ceramics/category_id/90/p/{}.html'
for i in range(1000):
    url = ceramic_base_url.format(i+1)
    if col_record.find({"url": url, "success": True}).count() > 0:
        print('已经爬取过第{}页'.format(i+1))
        continue

    print('开始爬取第{}页'.format(i+1))

    try:
        html = requests.get(url).text
        ceramic_s = etree.HTML(html)
        trs = ceramic_s.xpath("//div[@class='building2']//div[@class='table1']/table/tbody/tr")[1:]

        items = []
        for tr in trs:
            tds = tr.xpath('./td')
            item = {
                "name": tds[0].xpath('./a/text()')[0].strip(),
                "href": tds[0].xpath('.//@href')[0],
                "img_url": tds[0].xpath('.//@src')[0],
                "year": tds[1].xpath('./text()')[0].strip(),
                "type": tds[2].xpath('./text()')[0].strip(),
                "place": tds[3].xpath('./text()')[0].strip()
            }
            items.append(item)
        col.insert_many(items)
        col_record.insert({"url": url, "success": True})
    except Exception as e:
        print(e)
        col_record.insert({"url": url, "err": str(e)})