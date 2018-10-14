from pymongo import MongoClient
import requests
from lxml import etree

client = MongoClient()
db = client.dpm
col = db.paint
col_record = db.paint_downloaded

paints_base_url = 'http://www.dpm.org.cn/searchs/paints/category_id/91/p/{}.html'
for i in range(1000):
    url = paints_base_url.format(i+1)
    if col_record.find({"url": url, "success": True}).count() > 0:
        print('已经爬取过第{}页'.format(i+1))
        continue

    print('开始爬取第{}页'.format(i+1))

    try:
        html = requests.get(url).text
        s = etree.HTML(html)
        trs = s.xpath("//div[@class='building2']//div[@class='table1']/table/tbody/tr")
        if len(trs) < 1:
            print('结束')
            break

        trs = trs[1:]

        items = []
        for tr in trs:
            tds = tr.xpath('./td')
            item = {
                "name": tds[0].xpath('./a/text()')[0].strip(),
                "href": tds[0].xpath('.//@href')[0],
                "img_url": tds[0].xpath('.//@src')[0],
                "year": tds[1].xpath('./text()')[0].strip(),
                "type": tds[2].xpath('./text()')[0].strip(),
                "author": tds[3].xpath('./text()')[0].strip()
            }
            items.append(item)
        col.insert_many(items)
        col_record.insert({"url": url, "success": True})
    except Exception as e:
        print(e)
        col_record.insert({"url": url, "err": str(e), "success": False})