from pymongo import MongoClient

import requests
import os


class ImageDownloader(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.client = MongoClient()
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
            print("create dir", self.data_dir)
        self.db_name = None
        self.col_name = None
        self.db = None
        self.col = None
        self.recs = None
        self.record_db = None
        self.record_col = None
        self.url_key = None
        self.id_key = None
        self.start = None
        self.headers = None

    def set_db_info(self, db_name, col_name, record_db_name, record_col_name, url_key, id_key):
        self.db_name = db_name
        self.col_name = col_name
        self.db = self.client.get_database(db_name)
        self.col = self.db.get_collection(col_name)
        self.recs = list(self.col.find())
        self.record_db = self.client.get_database(record_db_name)
        self.record_col = self.record_db.get_collection(record_col_name)
        self.url_key = url_key
        self.id_key = id_key
        if not self.url_key in self.recs[0]:
            print("WRONG URL KEY, please check your db info.")
        if not self.id_key in self.recs[0]:
            print("WRONG ID KEY, please check your db info.")
        print("init db on ", self.db_name, self.col_name)
        print("init data_dir on", self.data_dir)

    def set_start(self, already_done_number):
        self.start = already_done_number // 100 * 100

    def set_headers(self, headers):
        self.headers = headers

    def check_response(self, url, response):
        if response.status_code != 200:
            print(url, "NOT 200 OK!")
            return False
        if not response.headers["Content-Type"].startswith("image"):
            print(url, "NOT IMAGE!")
            return False
        return True

    def write_and_record(self, url, response, f_dst):
        with open(f_dst, 'wb') as f:
            f.write(response.content)
            print("wrtie file ", f_dst)
        self.record_col.insert({"url": url, "success": True})

    def download_from_start(self):
        i = self.start
        for x in self.recs[self.start]:
            i += 1
            url = x[self.url_key]
            id = x[self.id_key]

            if self.record_col.find({"url": url, "success": True}).count() > 0:
                print(id, url, "已经下载")

            print("下载", id, url)
            try:
                res = requests.get(url, headers=self.headers)
                if not self.check_response(url, res):
                    continue
                f_dst = os.path.join(self.data_dir, "{}.jpg".format(id))
                self.write_and_record(url, res, f_dst)
            except Exception as e:
                print(id, url, e)


if __name__ == '__main__':
    test_data_dir = ""
    data_dir = ""
    headers = {}
    downloader = ImageDownloader(test_data_dir)
    downloader.set_db_info("metadata", "mma_img_to_medium",
                           "record", "mma_img_download", "url", "id")
    downloader.set_headers(headers)
    downloader.set_start(0)
    downloader.download_from_start()