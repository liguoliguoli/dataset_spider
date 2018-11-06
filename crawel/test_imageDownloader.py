from unittest import TestCase
from crawel.imgDownloader import ImageDownloader
import requests
import os


class TestImageDownloader(TestCase):

    def setUp(self):
        self.test_dir = ""
        self.test_headers = {}
        self.test_db_name = "test_db"
        self.test_col_name = "test_col"
        self.test_record_db_name = "test_record_db_anme"
        self.test_record_col_name = "test_record_col_name"
        self.test_url_key = "url_key"
        self.test_id_key = "id_key"
        self.test_start = 0
        self.test_url = ""
        self.test_file = "test.jpg"
        self.dl = ImageDownloader(self.test_dir)

    def test_set_db_info(self):
        dl = self.dl
        dl.set_db_info(self.test_db_name, self.test_col_name,
                       self.test_record_db_name, self.test_record_col_name,
                       self.test_url_key, self.test_id_key)
        self.assertTrue(isinstance(dl.recs, list))
        self.assertTrue(len(dl.recs) > 0)
        self.assertTrue(self.test_url_key in dl.recs[0])
        self.assertTrue(self.test_id_key in dl.recs[0])

    def test_set_start(self):
        dl = self.dl
        dl.set_start(self.test_start)
        self.assertEqual(self.test_start, dl.start)

    def test_set_headers(self):
        dl = self.dl
        dl.set_start(self.test_headers)
        self.assertEqual(self.test_headers, dl.headers)

    def test_check_response(self):
        dl = self.dl
        dl.set_headers(self.test_headers)
        res = requests.get(self.test_url, dl.headers)
        self.assertTrue(dl.check_response(res))

    def test_write_and_record(self):
        dl = self.dl
        res = requests.get(self.test_url, dl.headers)
        dl.set_db_info(self.test_db_name, self.test_col_name,
                       self.test_record_db_name, self.test_record_col_name,
                       self.test_url_key, self.test_id_key)
        dl.write_and_record(self.test_url, res, self.test_file)
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, self.test_file)))
        record_col = dl.record_col
        self.assertTrue(record_col.find({"url": self.test_url, "success": True}).count() > 0)

    def test_download_from_start(self):
        self.fail()
