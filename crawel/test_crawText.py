from unittest import TestCase
from crawel.craw_dpm_text import CrawText
from lxml import html


class TestCrawText(TestCase):

    def setUp(self):
        self.tc = CrawText()
        self.test_url = "http://www.dpm.org.cn/collection/ceramic/246386.html"
        self.headers = {
            "cookie": "UM_distinctid=1666828adaa462-0db21f7614f732-8383268-1fa400-1666828adab1bb; CNZZDATA1261553859=51641845-1539341757-null%7C1540440890; Hm_lvt_0934d73eb282e505cae957348c97af7c=1539345789,1539439313,1539781289,1540443942; PHPSESSID=3696d163d32ab06072e20fb0052649b9; saw_terminal=default; Hm_lpvt_0934d73eb282e505cae957348c97af7c=1540443970; cn_1261553859_dplus=%7B%22distinct_id%22%3A%20%221666828adaa462-0db21f7614f732-8383268-1fa400-1666828adab1bb%22%2C%22sp%22%3A%20%7B%22%24recent_outside_referrer%22%3A%20%22%24direct%22%2C%22%24_sessionid%22%3A%200%2C%22%24_sessionTime%22%3A%201540445068%2C%22%24dp%22%3A%200%2C%22%24_sessionPVTime%22%3A%201540445068%7D%2C%22initial_view_time%22%3A%20%221539341757%22%2C%22initial_referrer%22%3A%20%22https%3A%2F%2Fwww.google.com%2F%22%2C%22initial_referrer_domain%22%3A%20%22www.google.com%22%7D"
        }
        self.base_url = "http://www.dpm.org.cn"

    def test_get_res(self):
        r = self.tc.get_res(self.test_url, self.headers)
        self.assertEqual(r.status_code, 200)
        print(r.headers)
        # print(r.text)

    def test_extract_txt_from_html(self):
        r = self.tc.get_res(self.test_url, self.headers)
        h, p, a = self.tc.extract_txt_from_html(r.text)
        self.assertTrue(len(h) > 0)
        self.assertTrue(len(p) > 0)
        self.assertTrue(len(a) > 0)
        self.assertTrue(isinstance(h, str))
        self.assertTrue(isinstance(p, str))
        self.assertTrue(isinstance(a, str))
        print(h)
        print(p)
        print(a)

    def test_get_rec_from_db(self):
        rec = self.tc.get_rec_from_db("dpm_all", "search_res")
        self.assertTrue(isinstance(rec, list))
        self.assertTrue(len(rec) > 0)
        self.assertTrue("rec_id" in rec[0])
        self.assertTrue("href" in rec[0])

    def test_get_id2text(self):
        recs = self.tc.get_rec_from_db("dpm_all", "search_res")
        ids, texts, recs = self.tc.get_id2text(recs[:5], self.base_url, self.headers)
        self.assertTrue(len(ids) == len(texts))
        self.assertTrue(isinstance(ids[0], int))
        self.assertTrue(len(texts[0]) == 3)
        self.assertTrue(len(texts) == 5)
        self.assertTrue("text" in recs[0])
        print(texts[:2])
