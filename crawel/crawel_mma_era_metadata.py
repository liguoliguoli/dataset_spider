from pymongo import MongoClient
import requests

client = MongoClient()
meta_db = client.get_database("metadata")
meta_col = meta_db.get_collection("mma_facets")
eras = meta_col.find({"id": "era"})[0]["values"]

db_name = "mma_era"
mma_era_db = client.get_database(db_name)

headers = {
    "user-agent":"Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.96 Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    # "cookie":"visid_incap_1661922=uilOjJfxSTSpb6hRWraddjb/tVsAAAAAQUIPAAAAAACedP9EI32tyNibh5rTTd5Y; optimizelyEndUserId=oeu1538654013492r0.050263342420134416; _ga=GA1.2.278461422.1538654023; __qca=P0-547302917-1538654078337; visid_incap_1661977=/KjGl7b5RX+eHydZgJ5WIJz/tVsAAAAAQUIPAAAAAACncxtUDYJmeh5R7JLVWUO7; visid_incap_1662004=qTEOeSS5TJCGSTiGdwCQQPz/tVsAAAAAQUIPAAAAAAA6icggVpKHaBpVyBdke/sU; _gid=GA1.2.1937211985.1538827255; incap_ses_959_1661977=1AdJHSZHMz82WYxChw1PDQSkuFsAAAAAGC8bw8MI0ZHKn2Hh3TkNyw==; incap_ses_959_1661922=hpO6PSHxtTwfbpNChw1PDVaxuFsAAAAAJYxI0wEDy6gDy9OPTfh2NA==; incap_ses_959_1662004=YnSXdIw1yW+0b5NChw1PDVmxuFsAAAAA3LShvKVCvUD+WDqxxI/5uA==; _ceg.s=pg6hgw; _ceg.u=pg6hgw"
    "cookie":"SC_ANALYTICS_GLOBAL_COOKIE=b5afd0ce9ad74856abb0f5e2765047f0|False; visid_incap_1661922=4StfOrMoQ1qX59YbrLiEIvq1vFsAAAAAQUIPAAAAAADu1banOULod1zNTLjtNIjD; optimizelyEndUserId=oeu1539094012804r0.25007857242903153; _ga=GA1.2.1156854385.1539094013; __qca=P0-2015642789-1539094013418; ki_r=; ki_s=191527%3A0.0.0.0.0; visid_incap_1661977=C2dJXzrjTnSeBa/WOiW8pxS2vFsAAAAAQUIPAAAAAABvDB6gzpJLNCMAYix5uGW0; visid_incap_1662004=9FcOvXybQM66rIl1itJOsxe2vFsAAAAAQUIPAAAAAACV87sR+l8eBolcr0Fwk/ui; incap_ses_219_1661922=rx1La6zWnB3mcupdOw4KA0MNv1sAAAAAbrR82gOOr5JRq1HocYdQKw==; _gid=GA1.2.742836597.1539247431; ASP.NET_SessionId=ypf154zatn2l2bpmrwq0mzt5; __RequestVerificationToken=YNOQIlVXYDNOot6jgu6Y0Uu9pAfHbHx15bBrlwtYAgn2r6RHRX-sG3uHBw3r65RV7Nbe4OA2E6BrBJhYWrbO89PrH_U1; website#lang=en; __utma=85941765.1156854385.1539094013.1539247450.1539247450.1; __utmc=85941765; __utmz=85941765.1539247450.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmb=85941765.2.10.1539247450; __atuvc=2%7C41; __atuvs=5bbf0d59628f08c6001; _ceg.s=pgfeyh; _ceg.u=pgfeyh; incap_ses_219_1661977=RzoCPVt5yiEniupdOw4KA3gNv1sAAAAAKu2QVUKZ27REX4EdbOmddw==; ki_t=1539094015151%3B1539247480127%3B1539247488482%3B2%3B6; _gat_UA-72292701-1=1"
}

url_pattern = "https://www.metmuseum.org/api/collection/collectionlisting?artist=&department=&era={}&geolocation=&material=&offset={}&pageSize=0&perPage={}&searchField=All&showOnly=&sortBy=relevance&sortOrder=asc"
page_size = 100
for era in eras:
    col_name = era["name"]
    col = mma_era_db.get_collection(col_name)
    if col.count() < era["count"]:
        mma_era_db.drop_collection(col_name)
        print("开始下载{}".format(col_name))
        col = mma_era_db.get_collection(col_name)
        count = 0
        i = 0
        while count < era["count"]:
            url = url_pattern.format(era["label"].replace(" ",'+'), count, page_size)
            print(count)
            try:
                res = requests.get(url, headers=headers).json()
                col.insert_many(res["results"])
                count += len(res["results"])
            except Exception as e:
                print(e, url)

    else:
        print("{}下载完成".format(col_name))
        continue