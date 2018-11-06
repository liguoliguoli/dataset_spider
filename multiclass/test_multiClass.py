from unittest import TestCase
from multiclass.dpmDataset import MultiClassDataset
from keras.applications.vgg16 import VGG16
import numpy as np


class TestMultiClass(TestCase):

    def setUp(self):
        self.all_dir = "..\\..\\..\\img_precess\\img\\all"
        self.split_dir = "..\\..\\..\\img_precess\\img\\split"
        # self.data_dir = "I:\\img\\dpm\\dataset\\{}\\".format(self.split_dir.split("\\")[-2])
        self.dataset_dir = r"I:\img\dpm\dataset\type_split"
        self.mc = MultiClassDataset(self.split_dir, (224, 224))

    def test_set_model(self):
        self.mc.set_model(VGG16(weights="imagenet", input_shape=(224, 224, 3), include_top=False))
        self.assertEqual(self.mc.cnn_model.input_shape[1:], (224, 224, 3))
        self.assertEqual(self.mc.ins, (224, 224))

    def test_get_f_cnn(self):
        self.mc.set_model(VGG16(weights="imagenet", input_shape=(224, 224, 3), include_top=False))
        f_cnn = self.mc.get_f_cnn()
        self.assertTrue(isinstance(f_cnn, np.ndarray))
        self.assertEqual(f_cnn.shape[1:], (7, 7, 512))

    def test_get_f_hsv(self):
        f_hsv = self.mc.get_f_hsv()
        self.assertEqual(len(f_hsv[0]), 72)

    def test_get_id2t(self):
        id2t = self.mc.get_id2t()
        print(id2t)

    def test_get_label(self):
        l = self.mc.get_label()
        print(l)
        self.assertTrue(isinstance(l, np.ndarray))
        self.assertEqual(np.max(np.argmax(l), axis=-1), 1)
        self.assertEqual(l.shape[1:], (2,))

    def test_load_data(self):
        order, f_hsv, f_cnn = self.mc.load_data(self.dataset_dir)
        self.assertTrue(isinstance(order, dict))
        self.assertTrue(isinstance(order, dict))
        self.assertTrue(isinstance(order, dict))
        for atr in ["ids", "img_paths", "id2t", "labels", "label_set", "label_y"]:
            self.assertTrue(atr in order)

        ids = order["ids"]
        id2t = order["id2t"]
        img_paths = order["img_paths"]
        labels = order["labels"]
        label_y = order["label_y"]
        label_set = order["label_set"]

        f_cnn = f_cnn["f_cnn"]
        f_hsv = f_hsv["f_hsv"]
        self.assertTrue(isinstance(f_cnn, np.ndarray))
        self.assertTrue(isinstance(f_hsv, np.ndarray))

        self.assertEqual(len(ids), len(labels))
        self.assertEqual(len(ids), len(label_y))
        self.assertEqual(len(ids), len(img_paths))
        self.assertEqual(np.max(np.argmax(label_y, axis=-1)), len(label_set)-1)
        for i in np.random.randint(0, len(ids), 5):
            id = ids[i]
            label = labels[i]
            y = label_y[i]
            self.assertEqual(np.argmax(y), label_set.index(label))
            self.assertEqual(label, id2t[id])


