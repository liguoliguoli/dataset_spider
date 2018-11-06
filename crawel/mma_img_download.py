from pymongo import MongoClient

import requests
import os

meta_db_name = "metadata"
# meta_col_name = "mma_all_meta"
# mma_db_name = "crawel"

client = MongoClient()

meta_db = client.get_database(meta_db_name)

# meta_col = meta_db.get_collection(meta_col_name)

record_db = client.get_database("record")

record_col = record_db.get_collection("mma_img_download")

col = meta_db.get_collection("mma_img_to_medium")

# base_url = "https://www.metmuseum.org/art/collection/search#!?perPage=20&offset=0&pageSize=0&sortBy=relevance&sortOrder=asc&searchField=All"
#
# meta_url = "https://www.metmuseum.org/api/collection/collectionlisting?offset=0&pageSize=0&perPage=20&searchField=All&showOnly=&sortBy=relevance&sortOrder=asc"

headers = {
    "user-agent":"Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.96 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    # "cookie":"visid_incap_1661922=0HOmKS/GSaexKtuOaV2rraT6wVsAAAAAQUIPAAAAAADeAO9cLYXjViu//jOackB2; _ga=GA1.2.1790863583.1539439274; __qca=P0-1697327361-1539439275827; optimizelyEndUserId=oeu1539439279171r0.1713517101835651; visid_incap_1662004=YZ0oSWzfRVKpt2tFq5xCNLokw1sAAAAAQUIPAAAAAAB2XZJk07PQSfkqrP/wcFTZ; incap_ses_373_1662004=w42gR2Og82EYHLwx8SktBbokw1sAAAAA9q0Vv3ZWvMRwzSuwZbqdVQ==; visid_incap_1661977=tvxj/dwFTyaOMGdZmC03+9Qkw1sAAAAAQUIPAAAAAACs2xXY1w4rsNSsO5rORpP3; incap_ses_373_1661922=ejBPM9V4OFxko3Qz8SktBXSwxFsAAAAAlrSQK0b3PdbizvLI9Gr5Ew==; _gid=GA1.2.1876371791.1539616890; _dc_gtm_UA-72292701-1=1; _gat_UA-72292701-1=1; _ceg.s=pgnc0l; _ceg.u=pgnc0l; incap_ses_373_1661977=CsIgSQspB1fx0XQz8SktBZWwxFsAAAAAsynH9iCjrxonKl6NMpZzGw=="
    # "cookie":"visid_incap_1661922=VqDL94BHQWaj8VsDuMPpbRx7xVsAAAAAQUIPAAAAAADR/nIZHLvPDM63D5Eej4/f; optimizelyEndUserId=oeu1539668776031r0.6821406509246857; _ga=GA1.2.289133427.1539668778; __qca=P0-2044393074-1539668777688; visid_incap_1661977=jypr2k5rQfmUtp6ZnWB2B6x6xVsAAAAAQUIPAAAAAAC2mjVclyKoWhxKQZNlbb66; incap_ses_259_1661922=E8JzfSuhpHRsbg+ckCiYA+Xr01sAAAAAEw78yjxRF6VRZOK0dogTIA==; _gid=GA1.2.1681681316.1540615147; _dc_gtm_UA-72292701-1=1; _fbp=fb.1.1540615147762.70722539; _gat_UA-72292701-1=1; _ceg.s=ph8q9r; _ceg.u=ph8q9r; incap_ses_259_1661977=XPzGdAoRlAn5gw+ckCiYA//r01sAAAAAwze6DbeO0420FK/eJrPhgQ=="
    "cookie":"visid_incap_1661922=/3jWfS/aQWqk3qkHNt1dzPso1VsAAAAAQUIPAAAAAADVkoLB/tsENC09b79e2uUX; incap_ses_259_1661922=xLZvaS7ZYBZ041mdkCiYA/so1VsAAAAAj6A+ShoOkEb7lo0K+AatoA==; optimizelyEndUserId=oeu1540696316586r0.448068460585052; _ga=GA1.2.1192052307.1540696317; _gid=GA1.2.438162514.1540696317; _dc_gtm_UA-72292701-1=1; _fbp=fb.1.1540696317618.587886240; __qca=P0-9370698-1540696317585; _gat_UA-72292701-1=1; _ceg.s=phagwm; _ceg.u=phagwm; visid_incap_1661977=Uuse0lEnRCGWm3qmztMMhhgp1VsAAAAAQUIPAAAAAAD71jZZ3LTnxAXLePYY3hAp; incap_ses_259_1661977=RjUsUjus1S6c8lmdkCiYAxgp1VsAAAAAvhYoSIHZjMt0cPwk8ESAQQ==; visid_incap_1662004=8w+2sNeGTrCP+U6w1pzsexop1VsAAAAAQUIPAAAAAADbFYscMW3K9yRPbZ1P24Ji; incap_ses_259_1662004=2b43QQyUmVbj81mdkCiYAxsp1VsAAAAAJZ98AVmx1IK8jVDG6edt3A=="
}

data_dir = r"I:\img\mma\origin\mediums_100"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
n = 0
i = n
for x in col.find()[n:]:
    i += 1
    url = x["url"]
    if record_col.find({"url": url, "success": True}).count() > 0:
        if i % 1000 == 0:
            print(i, url, "已经下载")
        continue
    print("下载", i, url)
    try:
        res = requests.get(url,headers=headers)
        if not res.headers["Content-Type"].startswith("image"):
            record_col.insert({"url": url, "success": False, "content_type": res.headers["Content-Type"]})
            print("非图片类型", res.headers["Content-Type"])
            continue
        data = res.content
        with open("{}\\{}.jpg".format(data_dir, i), 'wb') as f_img:
            f_img.write(data)
            record_col.insert({"url": url, "success": True})
    except Exception as e:
        print(e)