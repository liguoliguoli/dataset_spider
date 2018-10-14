import requests
from lxml import etree
from pymongo import MongoClient,ASCENDING
client = MongoClient()
db = client.test_db
collection = db.taibeigugong_opendata_img
collection.create_index([("id", ASCENDING)], unique=True)


"""
修改：http url请求出错，需要记录下来是哪个url出错，方便后续再次请求
"""

nb_pages = 540
for i in range(nb_pages)[1:]:
    url = 'https://theme.npm.edu.tw/opendata/DigitImageSets.aspx?pageNo={}'.format(i)
    print('准备获取第{}页，url：{}'.format(i, url))
    # handle http exception and xpath exception
    try:
        res = requests.get(url)
        html = res.text
        s = etree.HTML(html)
        li_lists = s.xpath('//ul[@class="photoList"]/li')
        for li in li_lists:
            title = li.xpath('.//div[@class="photoText"]/text()')[0]
            # title = li.xpath('.//@alt')[0] sometimes got null result
            print(title)
            high_solution_img_download_url = li.xpath('.//@href')[0]
            print(high_solution_img_download_url)
            low_solution_img_download_url = li.xpath('.//@src')[0]
            print(low_solution_img_download_url)
            id = high_solution_img_download_url.split('=')[1]
            print(id)

            # handle insert duplicate exception
            try:
                collection.insert(
                    {
                        "id": id,
                        "name": title,
                        "high_solution_img_download_url": high_solution_img_download_url,
                        "low_solution_img_download_url": low_solution_img_download_url
                    }
                )
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
