import numpy as np
from imutils import paths
from img_precess.featureExtracter import FeatureExtractor
import os
import pickle
from datetime import datetime


class MultiClassDataset(object):
    """
    class for create feature dataset to disk, and those feature have the save sample order
    """

    def __init__(self, datadir):
        self.extractor = FeatureExtractor()
        self.data_dir = datadir
        self.img_paths = list(paths.list_images(self.data_dir))
        self.img_paths.sort()
        self.id2t = None
        self.ids = None
        self.labels = None
        self.label_set = None
        self.label_y = None
        self.f_cnn = None
        self.f_hsv = None
        self.normed = None

        self._set_label()

    def _set_label(self):
        """

        :return:ids,labels and labels_Y
        """
        self.id2t = self.extractor.get_id2t_from_db()
        self.ids = self.extractor.get_ids_from_paths(self.img_paths)

        labels = []
        for id in self.ids:
            labels.append(self.id2t[id])
        self.labels = labels

        self.label_set = list(set(self.labels))

        label_y = []
        for label in self.labels:
            label_y.append(self.label_set.index(label))
        label_y = np.asarray(label_y)
        # label_y = keras.utils.to_categorical(label_y)
        self.label_y = label_y

        print("data dir:", self.data_dir, "len of image_paths:", len(self.img_paths),
              "nb_sample:", len(self.ids), "nb_label:", len(self.label_set))

    def get_normed(self, ins):
        """
        return data normed like (224, 224, 3) , value in [0,1]
        :return:
        """
        self.normed = self.extractor.extract_normed_from_disk(self.img_paths, ins)
        print("get normed data:", len(self.normed))
        return self.normed

    def get_f_cnn(self, cnn_model, ins):
        """
        use cnn model to extract cnn
        :return:
        """
        self.get_normed(ins)
        print(cnn_model.input_shape, self.normed.shape)
        assert cnn_model.input_shape[1:] == self.normed.shape[1:]
        self.f_cnn = self.extractor.extract_cnn(self.normed, cnn_model)
        print("get f_cnn:", len(self.f_cnn))
        return self.f_cnn

    def get_f_hsv(self):
        """
        return hsv feature
        :return:
        """
        self.f_hsv = self.extractor.extract_hsv(self.img_paths)
        print("get f_hsv:", len(self.f_hsv))
        return self.f_hsv

    def get_f_lbp(self):
        """

        :return:
        """
        self.f_lbp = self.extractor.extract_lbp(self.img_paths)
        print("get f_lbp:", len(self.f_lbp))
        return self.f_lbp

    def get_f_hu(self):
        """

        :return:
        """
        self.f_hu = self.extractor.extract_hu(self.img_paths)
        print("get f_hu:", len(self.f_hu))
        return self.f_hu

    def get_f_hog(self):
        """

        :return:
        """
        self.f_hog = self.extractor.extract_hog(self.img_paths)
        print("get f_hog:", len(self.f_hog))
        return self.f_hog

    def get_f_word_sequences(self, word_dict_size, padding_length):
        """
        return dt_texts, sequences, padding_seq
        :param word_dict_size:
        :param padding_length:
        :return:
        """
        id2nout = self.extractor.get_id2noun_from_db()
        all_texts = [x["nout_text"] for x in id2nout]

        id2noun_mp = {}
        for x in id2nout:
            id2noun_mp[x["img_id"]] = x["nout_text"]
        dt_texts = []
        for img_id in self.ids:
            dt_texts.append(id2noun_mp[img_id])
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        tokenizer = Tokenizer(num_words=word_dict_size, char_level=False, split=' ')
        tokenizer.fit_on_texts(all_texts)
        word_dict = list(tokenizer.word_index.items())
        word_dict.sort(key=lambda x: x[1])
        print(word_dict[:100])
        sequences = tokenizer.texts_to_sequences(dt_texts)
        padding_seq = pad_sequences(sequences, maxlen=padding_length)
        return dt_texts, sequences, padding_seq


def dump(feature, dts, cnn_model=None, desc_len=200, desc_dict_size=50000, save_dir=r"I:\img\dpm\dataset\type_enhanced_1p3_split_tvt"):
    """
    dump trn,val,tst
    :param feature:
    :param cnn_model:
    :param desc_len:
    :param desc_dict_size:
    :param save_dir
    :return:
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("DUMP %s:" % feature)
    name = ["trn", "val", "tst"]
    data = None
    t = datetime.now()
    for i, dt in enumerate(dts):
        print("start %s" % name[i])
        if cnn_model:
            data = dt.get_f_cnn(cnn_model, (224, 224))
        elif feature == "hu":
            data = dt.get_f_hu()
        elif feature == "lbp":
            data = dt.get_f_lbp()
        elif feature == "hsv":
            data = dt.get_f_hsv()
        elif feature == "hog":
            data = dt.get_f_hog()
        elif feature.startswith("desc"):
            _, _, data = dt.get_f_word_sequences(desc_dict_size, desc_len)

        with open(os.path.join(save_dir, "%s_%s" % (name[i], feature)), 'wb') as f:
                pickle.dump({"data": data, "y": dt.label_y, "label_set": dt.label_set}, f)
    print("done")
    print(datetime.now() - t)


def load(feature, save_dir=r"I:\img\dpm\dataset\type_enhanced_1p3_split_tvt"):
    """
    return trn,val,tst
    :param feature:
    :param save_dir
    :return:
    """
    print("LOAD %s:" % feature)
    name = ["trn", "val", "tst"]
    t = datetime.now()
    data = []
    for i in range(3):
        with open(os.path.join(save_dir, "%s_%s" % (name[i], feature)), 'rb') as f:
            data.append(pickle.load(f))
    # print("done")
    print(datetime.now() - t)
    return data


if __name__ == '__main__':

    #load all for test

    trn_y = []
    val_y = []
    tst_y = []
    for f in ["hog", "lbp", "resnet50", "hu", "resnet50_ft", "desc_1000_100", "hsv"]:
        x, y, z = load(f)
        trn_y.append(x["y"])
        val_y.append(y["y"])
        tst_y.append(z["y"])
    trn_y = np.asarray(trn_y)
    val_y = np.asarray(val_y)
    tst_y = np.asarray(tst_y)
    assert np.all(trn_y[0] == trn_y[1])
    trn_y -= trn_y[0]
    val_y -= val_y[0]
    tst_y -= tst_y[0]
    print(np.sum(np.abs(trn_y)))


    # dump all

    # base_dir = r"I:\img\dpm\origin\type_enhanced_1p3_split_tvt"
    # train_dir, val_dir, test_dir = [os.path.join(base_dir, x) for x in ["train", "val", "test"]]
    # train_dt = MultiClassDataset(train_dir)
    # val_dt = MultiClassDataset(val_dir)
    # test_dt = MultiClassDataset(test_dir)
    # dts = [train_dt, val_dt, test_dt]
    #
    # trained_mdoel = keras.models.load_model(r"I:\img\model\test4\2\resnet50_1\resnet50_1")
    # flatten_layer = trained_mdoel.get_layer("flatten_2")
    # cnn_model2 = keras.models.Model(trained_mdoel.input, flatten_layer.output)
    #
    # dump("hog", dts)
    # dump("lbp", dts)
    # dump("hu", dts)
    # dump("resnet50", dts, cnn_model=ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))
    # dump("resnet50_ft", dts,  cnn_model=cnn_model2)
    # dump("desc_1000_100", dts, desc_dict_size=1000, desc_len=100)
    # dump("hsv", dts)