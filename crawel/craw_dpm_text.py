from pymongo import MongoClient
import os
import pickle
import requests
from lxml import html


class CrawText(object):

    def get_res(self, url, headers):
        """

        :param url:
        :param headers:
        :return:
        """
        try:
            r = requests.get(url, headers)
            return r
        except Exception as e:
            print(url, e)
            raise Exception

    def extract_txt_from_html(self, text):
        """

        :param text:  response html
        :return:
        """
        s = html.fromstring(text)
        text_div = s.xpath("//div[@id='hl_content']//div[@class='text']")[0]
        h = text_div.xpath("./h3/text()")
        p = text_div.xpath(".//p/text()")
        a = text_div.xpath(".//a/text()")
        h = [x.strip() for x in h]
        p = [x.strip() for x in p]
        a = [x.strip() for x in a]
        h = " ".join(h)
        p = " ".join(p)
        a = "".join(a)
        return h, p, a

    def get_rec_from_db(self, db_name, col_name):
        """
        get metadata from db,col
        :param db_name:
        :param col_name:
        :return:
        """
        client = MongoClient()
        db = client.get_database(db_name)
        col = db.get_collection(col_name)
        return list(col.find())

    def get_id2text(self, recs, base_url, headers):
        """
        get rec ids ,and corrspondind describe texts from rec["href"]
        :param recs:
        :param base_url:
        :param headers:
        :return:
        """
        ids = []
        texts = []
        for i, x in enumerate(recs):
            try:
                rec_id = x["rec_id"]
                href = x["href"]
                url = base_url + href
                print(i, "get data from: ", url)
                r = self.get_res(url, headers)
                h, p, a = self.extract_txt_from_html(r.text)
                ids.append(rec_id)
                texts.append((h, p, a))
                x["text"] = (h, p, a)
                print(h, p, a)
            except Exception as e:
                print(url, e)
        return ids, texts, recs


if __name__ == '__main__':
    tc = CrawText()
    headers = {
        "cookie": "UM_distinctid=1666828adaa462-0db21f7614f732-8383268-1fa400-1666828adab1bb; CNZZDATA1261553859=51641845-1539341757-null%7C1540440890; Hm_lvt_0934d73eb282e505cae957348c97af7c=1539345789,1539439313,1539781289,1540443942; PHPSESSID=3696d163d32ab06072e20fb0052649b9; saw_terminal=default; Hm_lpvt_0934d73eb282e505cae957348c97af7c=1540443970; cn_1261553859_dplus=%7B%22distinct_id%22%3A%20%221666828adaa462-0db21f7614f732-8383268-1fa400-1666828adab1bb%22%2C%22sp%22%3A%20%7B%22%24recent_outside_referrer%22%3A%20%22%24direct%22%2C%22%24_sessionid%22%3A%200%2C%22%24_sessionTime%22%3A%201540445068%2C%22%24dp%22%3A%200%2C%22%24_sessionPVTime%22%3A%201540445068%7D%2C%22initial_view_time%22%3A%20%221539341757%22%2C%22initial_referrer%22%3A%20%22https%3A%2F%2Fwww.google.com%2F%22%2C%22initial_referrer_domain%22%3A%20%22www.google.com%22%7D"
    }
    base_url = "http://www.dpm.org.cn"
    recs = tc.get_rec_from_db("dpm_all", "search_res")
    _, _, new_recs = tc.get_id2text(recs, base_url, headers)
    client = MongoClient()
    db = client.get_database("dpm_all")
    col = db.get_collection("rec_with_desc")
    col.insert_many(new_recs)



