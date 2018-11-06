from unittest import TestCase
from img_precess.imagePreprocessor import ImagePrecessor
import os
from imutils import paths
import numpy as np


class TestImagePrecessor(TestCase):

    def setUp(self):
        self.processor = ImagePrecessor()
        self.test_dir = "..\\img\\all"
        self.dst_dir = "..\\img\\test"

    def tearDown(self):
        print("done")

    def test_crop_rectangle(self):
        print(os.getcwd())
        f_in = "..\\img\\in.jpg"
        f_out = "..\\img\\out.jpg"
        self.processor.crop_rectangle(f_in, f_out, 60)
        self.assertTrue(os.path.exists(f_out))

    def test_remove_noise(self):
        self.fail()

    def test_enhance_contrast(self):
        self.fail()

    def test_create_subdir(self):
        self.fail()

    def test_ensure_dir(self):
        self.fail()

    def test_create_train_val_test(self):
        self.fail()

    def test_crop(self):
        self.processor.crop(self.test_dir, self.dst_dir)
        self.assertTrue(os.path.exists(self.dst_dir))
        self.assertEqual(len(list(paths.list_images(self.test_dir))),
                         len(list(paths.list_images(self.dst_dir))))

    def test_blur(self):
        self.processor.blur(self.test_dir, self.dst_dir)
        self.assertTrue(os.path.exists(self.dst_dir))
        self.assertEqual(len(list(paths.list_images(self.test_dir))),
                         len(list(paths.list_images(self.dst_dir))))

    def test_enhance(self):
        self.processor.enhance(self.test_dir, self.dst_dir)
        self.assertTrue(os.path.exists(self.dst_dir))
        self.assertEqual(len(list(paths.list_images(self.test_dir))),
                         len(list(paths.list_images(self.dst_dir))))
