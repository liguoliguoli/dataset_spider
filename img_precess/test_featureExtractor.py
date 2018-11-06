from unittest import TestCase
from img_precess.featureExtracter import FeatureExtractor
import os
from imutils import paths
import numpy as np
from matplotlib import pyplot as plt


class TestFeatureExtractor(TestCase):

    def setUp(self):
        self.extractor = FeatureExtractor()
        self.all_dir = "..\\img\\all"
        self.split_dir = "..\\img\\split"
        self.split_img_paths = list(paths.list_images(self.split_dir))
        self.test_txt = "金镶 戒指，清，通长2.2 cm，径2.1cm。 红、蓝宝石都是以色命名的刚玉类宝石，属三方晶系，单晶呈柱状、桶状或近似腰鼓状，化学成分为氧化铝，硬度为9，密度为3.95-4.40，有透明、半透明、不透明三类，颜色以鸽子红、石榴红最为名贵。主要产自东南亚、印度、澳大利亚、巴西等地。古代，红宝石一直被用作护身符、辟邪符和装饰品。我国清代，红宝石还被用作亲王及一品官的顶戴标志。"

    def tearDown(self):
        print("done")

    def test_extract_hsv(self):
        hsvs = self.extractor.extract_hsv(self.split_img_paths)
        self.assertTrue(isinstance(hsvs, np.ndarray))
        self.assertEqual(len(hsvs[0]), 72)

    def test_extract_normed_from_disk(self):
        ims = self.extractor.extract_normed_from_disk(self.split_img_paths, (224, 224))
        self.assertTrue(np.all(ims[0] <= 1) and np.all(ims[0] >= 0))

    def test_get_ids_from_paths(self):
        res = self.extractor.get_ids_from_paths(self.split_img_paths)
        self.assertTrue(isinstance(res[0], int))
        print(self.split_img_paths)
        print(res)

    def test_normed_and_hsv(self):
        hsvs = self.extractor.extract_hsv(self.split_img_paths)
        ims = self.extractor.extract_normed_from_disk(self.split_img_paths, (224, 224))
        self.assertTrue(len(hsvs) == len(ims))

    def test_extract_cnn(self):
        from keras.applications.vgg16 import VGG16
        model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        normed = self.extractor.extract_normed_from_disk(self.split_img_paths, (224, 224))
        f = self.extractor.extract_cnn(normed, model)
        self.assertTrue(isinstance(f, np.ndarray))
        self.assertTrue(len(f) == len(normed))
        o_s = model.output_shape
        self.assertEqual(f.shape, (len(normed), o_s[1], o_s[2], o_s[3]))

    def test_extract_hu(self):
        f_hu = self.extractor.extract_hu(self.split_img_paths)
        self.assertTrue(isinstance(f_hu, np.ndarray))
        self.assertEqual(f_hu.shape[1], 7)
        print(f_hu)

    def test_extract_lbp(self):
        lbps = self.extractor.extract_lbp(self.split_img_paths)
        print(lbps)
        self.assertEqual(lbps.shape, (4, 26))

    def test_get_id2t_from_db(self):
        id2t = self.extractor.get_id2t_from_db()
        self.assertTrue(isinstance(id2t, dict))
        print(id2t)
        self.assertTrue(id2t[9] == "bamboos")
        self.assertTrue(id2t[14] == "bronzes")

    def test_extract_single_hog(self):
        et = self.extractor
        hogs = et.extract_single_hog()
        print(hogs)
        print(np.asarray(hogs).shape)
        # plt.imshow(hogs)
        # plt.show()

    def test_get_texts_from_ids(self):
        et = self.extractor
        texts = et.get_clear_texts_from_ids(range(7000))
        self.assertTrue(texts[14].startswith("清乾隆"))

    def test_get_origin_texts_from_ids(self):
        et = self.extractor
        texts = et.get_origin_texts_from_ids(range(7000))
        print(texts[36])
        self.assertTrue(texts[36].rstrip().startswith("韩子思"))

    def test_get_jiebacuttext_from_ids(self):
        et = self.extractor
        cut_texts = et.get_jiebacuttext_from_ids(range(10))
        print(cut_texts[1])

    def test_get_noun_from_text(self):
        et = self.extractor
        print(et.get_noun_from_text(self.test_txt))
        print(et.get_jieba_cut_clear_text(self.test_txt))

    def test_get_jieba_cut_clear_text(self):
        et = self.extractor
        print(et.get_jieba_cut_clear_text(self.test_txt))

    def test_get_nountext_from_ids(self):
        et = self.extractor
        texts = et.get_nountext_from_ids(range(10))
        print(texts[1])
