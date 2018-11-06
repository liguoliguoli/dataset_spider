from unittest import TestCase
from multiclass.dpmDataset import MultiClassDataset
import os

class TestMultiClassDataset(TestCase):

    def setUp(self):
        base_dir = r"I:\img\dpm\origin\type_enhanced_1p3_split_tvt"

        train_dir, val_dir, test_dir = [os.path.join(base_dir, x) for x in ["train", "val", "test"]]
        self.dt = MultiClassDataset(train_dir)

    def test_get_f_word_sequences(self):
        texts, _, padding = self.dt.get_f_word_sequences(1000, 50)
        print(texts[8])
        print(padding[8])

