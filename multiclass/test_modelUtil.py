from unittest import TestCase
from multiclass.modelUtil import ModelUtil


class TestModelUtil(TestCase):

    def setUp(self):
        self.mu = ModelUtil()

    def test_get_top_mlp(self):
        mlp = self.mu.get_top_mlp(256, 2, (7, 7, 512))
        mlp.summary()

    def test_get_plain_mlp(self):
        mlp = self.mu.get_plain_mlp(36, 3, 72)
        mlp.summary()
