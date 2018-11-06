from unittest import TestCase
from multiclass.transferLearner import TransferLearner
from keras.applications import ResNet50
import numpy as np
from pymongo import MongoClient
import os


class TestTransferLearner(TestCase):

    def setUp(self):
        self.test_data_dir = r"I:\img\test_dir"
        self.test_dst_dir = r"I:\img\model\test"
        self.model_name = "resnet50"
        self.mark = "resnet50_test"
        self.target_size = 224
        self.cnn_model = ResNet50(weights="imagenet", include_top=False,
                               input_shape=(self.target_size, self.target_size, 3))
        self.tl = TransferLearner(self.cnn_model, self.test_data_dir, self.test_dst_dir,
                    self.model_name, self.mark, target_size=(self.target_size, self.target_size))
        self.freezing_layer = "add_8"
        self.freezing_layer_index = 89
        # self.logger = Logger("..\\..\\..\\logs\\test.log").logger

    def test_init(self):
        self.assertTrue(self.tl.nb == 16)
        self.assertTrue(isinstance(self.tl.train_imgs, list))
        self.assertTrue(os.path.exists(os.path.join(self.test_dst_dir, self.mark)))
        self.assertTrue(os.path.isdir(os.path.join(self.test_dst_dir, self.mark)))

    def test_set_composed_model(self):
        tl = self.tl
        tl.set_composed_model()
        self.assertTrue(tl.composed_model.output_shape[1] == 16)

    def test_get_index_of_layer(self):
        tl = self.tl
        idx, layer_names = tl.get_index_of_layer(self.freezing_layer)
        print(idx)
        for i, n in enumerate(layer_names):
            print(i, n)

    def test_plot_hist(self):
        fake_hist = {
            "acc": np.arange(1, 20, 1),
            "val_acc": np.arange(0, 19, 1),
            "loss": np.arange(20, 1, -1),
            "val_loss": np.arange(21, 2, -1)
        }
        self.tl.plot_hist(fake_hist, "fake hist for test")

    def test_freezing_layers(self):
        tl = self.tl
        tl.set_composed_model()
        for i, layer in enumerate(tl.cnn_model.layers):
            print(i, layer.name, layer.trainable)
        tl.freezing_layers(89)
        for i, layer in enumerate(tl.cnn_model.layers):
            print(i, layer.name, layer.trainable)

    def test_transfer_learning(self):
        tl = self.tl
        tl.transfer_learning(epoch=1)
        times = tl.times
        hists = tl.hists
        scores = tl.scores
        epoches = tl.epoches
        lrs = tl.lrs
        print(times)
        print(hists)
        print(scores)
        print(epoches)
        print(lrs)
        self.assertTrue(len(times)==1)
        self.assertTrue(len(scores)==1)
        self.assertTrue(len(hists)==1)
        self.assertTrue(len(epoches) == 1)
        self.assertTrue(len(lrs) == 1)
        tl.plot_hist(hists[0], "test_transfer_learning")

    def test_get_score(self):
        pass

    def test_fine_tuning(self):
        tl = self.tl
        tl.transfer_learning(epoch=1)
        tl.fine_tuning(self.freezing_layer_index, epoch=1)
        times = tl.times
        hists = tl.hists
        scores = tl.scores
        epoches = tl.epoches
        lrs = tl.lrs
        print(times)
        print(hists)
        print(scores)
        print(epoches)
        print(lrs)
        self.assertTrue(len(times) == 2)
        self.assertTrue(len(scores) == 2)
        self.assertTrue(len(hists) == 2)
        self.assertTrue(len(epoches) == 2)
        self.assertTrue(len(lrs) == 2)
        self.tl.plot_hist(hists[0], "transfer learning train hist")

    def test_save(self):
        client = MongoClient()
        db = client.get_database("keras")
        col = db.get_collection("tf_compare_res")
        tl = self.tl
        tl.transfer_learning(epoch=1)
        tl.fine_tuning(self.freezing_layer_index, epoch=1)
        tl.save(comment="comment for test ")
        self.assertTrue(os.path.isdir(os.path.join(self.test_dst_dir, self.mark)))
        self.assertEqual(len(list(os.listdir(os.path.join(self.test_dst_dir, self.mark)))), 3)
        rec = list(col.find())[-1]
        self.assertEqual(rec["comment"], "comment for test ")
        self.assertEqual(rec["mark"], self.mark)
        self.assertEqual(rec["data_dir"], )

