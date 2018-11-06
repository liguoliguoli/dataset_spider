import cv2
import shutil
import numpy as np
import os
from PIL import Image, ImageEnhance
from datetime import datetime
from pymongo import MongoClient
import imutils
import pickle
from imutils import paths
from skimage import feature
import numpy as np
from matplotlib import pyplot as plt
from utils.db_tool import get_col


class FeatureExtractor(object):

    # def extract_raw_from_disk(self, img_paths):
    #     """
    #     :param img_paths: a list of img paths
    #     :return: list of Image object according to img_paths
    #     """
    #     t = datetime.now()
    #     imgs = []
    #     print("extract raw...")
    #     for i, p in enumerate(img_paths):
    #         print(i, "load ", p)
    #         try:
    #             im = Image.open(p)
    #             im = im.convert("RGB")
    #             imgs.append(im)
    #         except Exception as e:
    #             print(p, e)
    #     print(datetime.now() - t)
    #     return img_paths, imgs

    def extract_normed_from_disk(self, img_paths, shape):
        """
        :param img_paths:a list of img_paths
        :param shape: resize shape
        :return: np array X resized to (shape,shape,3) and scaled by 1./255, no Y
        """
        t = datetime.now()
        imgs = []
        img_ids = []
        print("extract normed from disk...")
        for i, p in enumerate(img_paths):
            print(i, "load ", p)
            try:
                im = Image.open(p)
                im = im.convert("RGB")
                im = im.resize(shape)
                im = np.asarray(im)
                imgs.append(im)
                img_ids.append(int(p.split("\\")[-1].split(".")[0]))
            except Exception as e:
                print(p, e)
        imgs = np.asarray(imgs, dtype=np.float32)
        imgs *= 1./255
        print(datetime.now() - t)
        return imgs

    # def extarct_normed_from_raw(self, raw_imset, shape):
    #     """
    #     use keras.preprocessing.image.ImageGenerator to normnize raw_imset
    #     :param raw_imset: raw_im Image object with different im shape
    #     :param shape: resize shape
    #     :return: np array X resized to (shape,shape,3) and scaled by 1./255, no Y
    #     """
    #     from keras.preprocessing.image import ImageDataGenerator
    #     datagen = ImageDataGenerator(rescale=1. / 255)
    #     gen = datagen.flow(raw_imset, target_size=(224, 224), shuffle=False, batch_size=32,
    #                        class_mode="none")
    #     X = []
    #     i = 0
    #     n = 6672
    #
    #     t1 = datetime.now()
    #     for a in gen:
    #         X.extend(a)
    #         i += len(a)
    #         print(i)
    #         if i >= n:
    #             break
    #     print(datetime.now() - t1)
    #
    #     X = np.asarray(X)
    #     print(X.shape)

    def get_ids_from_paths(self, img_paths):
        """
        from img_path to infer img type
        :param img_paths: like basedir\class_name\img_id.jpg
        :return: map {img_id: type}
        """
        ids = []
        for path in img_paths:
            t, i = path.split("\\")[-2], path.split("\\")[-1]
            id = int(i.split(".")[0])
            ids.append(id)
        return ids

    def get_id2t_from_db(self):
        """
        :return:
        """
        id2t = {}
        client = MongoClient()
        db = client.get_database("dpm_all")
        col = db.get_collection("type_ids")
        for x in col.find():
            t = x["type"]
            for id in x["img_ids"]:
                id2t[id] = t
        return id2t

    def extract_cnn(self, ims, model, bs=32):
        """
        use model to extract cnn features
        :param ims: normed im np array
        :param model: xx
        :param bs: batch_szie
        :return:
        """
        t = datetime.now()
        features = []
        i = 0
        print("extract f_cnn...")
        while i < len(ims):
            f = model.predict(ims[i: i + bs])
            features.extend(f)
            print("CNN", i)
            i += bs
        features = np.asarray(features)
        print(datetime.now() - t)
        return features

    def extract_hsv(self, img_paths, plot=False):
        """
        extract (72,) hsv hist feature from im
        :param img_paths: a list of target img paths
        :param plot: if true, plot origin image and hist
        :return: a list of (72,) hsv hist according to img_paths, and img_ids
        """

        def quanlize_H(v):
            hlist = [20, 40, 75, 155, 190, 270, 290, 316, 360]
            for i in range(9):
                if v <= hlist[i]:
                    h = i % 8
                    break
            return h

        def quanlize_SV(v):
            svlist = [21, 178, 255]
            for i in range(9):
                if v <= svlist[i]:
                    sv = i
                    break
            return sv

        q_h = np.vectorize(quanlize_H, otypes=[np.uint8])
        q_sv = np.vectorize(quanlize_SV, otypes=[np.uint8])

        hsvs = []
        ids = []
        t1 = datetime.now()

        print("extract hsv...")
        for i, im in enumerate(img_paths):
            img = cv2.imread(im)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            H, S, V = cv2.split(hsv)
            H *= 2

            t2 = datetime.now()
            H = q_h(H) * 9
            S = q_sv(S) * 3
            V = q_sv(V)
            nhsv = H + S + V
            print("HSV", i, datetime.now() - t2)
            hist = cv2.calcHist([nhsv], [0], None, [72],
                                [0, 72])  # 40x faster than np.histogramfaster than np.histogram
            if plot:
                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title("origin image")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.plot(hist, color='r')
                plt.title("HSV histgram")
                plt.xlim([0, 72])
                plt.show()

            hsvs.append(hist)
            ids.append(int(im.split("\\")[-1].split(".")[0]))

        hsvs = np.asarray(hsvs)
        hsvs = np.squeeze(hsvs, axis=-1)
        print(datetime.now() - t1)
        return hsvs

    def extract_hu(self, img_paths, threshhold=60):
        """
        extract hu moments for img in img_paths
        :param img_paths: list of img path
        :param threshhold: for binary threshhold
        :return: HU np array (n_samlpe, 7)
        """
        hus = []
        t = datetime.now()
        for i, im in enumerate(img_paths):
            t1 = datetime.now()
            image = cv2.imread(im, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
            gradient = cv2.subtract(gradX, gradY)
            gradient = cv2.convertScaleAbs(gradient)

            blurred = cv2.blur(gradient, (9, 9))
            (_, thresh) = cv2.threshold(blurred, threshhold, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            closed = cv2.erode(closed, None, iterations=4)
            closed = cv2.dilate(closed, None, iterations=12)

            _, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mask = np.zeros(image.shape).astype(image.dtype)
            color = [255, 255, 255]
            cv2.fillPoly(mask, cnts, color)

            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(gray_mask)
            hu_moments = cv2.HuMoments(moments)
            print("HU", i, datetime.now() - t1)
            hus.append(hu_moments)
        hus = np.asarray(hus)
        hus = np.squeeze(hus, axis=-1)
        print(datetime.now() - t)
        return hus

    def extract_lbp(self, img_paths, numPoints=24, radius=8, eps=1e-7):
        f_lbps = []
        t = datetime.now()
        for i, im in enumerate(img_paths):
            t1 = datetime.now()
            image = cv2.imread(im)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(gray, numPoints,
                                               radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, numPoints + 3),
                                     range=(0, numPoints + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            f_lbps.append(hist)
            print("LBP", i, datetime.now() - t1)
        f_lbps = np.asarray(f_lbps)
        print(datetime.now() - t)
        return f_lbps

    def extract_hog(self, img_paths):
        f_hogs = []
        t = datetime.now()
        for i, im in enumerate(img_paths):
            t1 = datetime.now()
            image = Image.open(im)
            image = image.resize((224, 224))
            features = feature.hog(image,  # input image
                                   orientations=9,  # number of bins
                                   pixels_per_cell=(20, 20),  # pixel per cell
                                   cells_per_block=(2, 2),  # cells per blcok
                                   block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                                   transform_sqrt=True,  # power law compression (also known as gamma correction)
                                   feature_vector=True,  # flatten the final vectors
                                   visualise=False)  # return HOG map
            f_hogs.append(features)
            print("HOG", i, datetime.now() - t1)
        f_hogs = np.asarray(f_hogs)
        print(datetime.now() - t)
        return f_hogs

    def get_jieba_cut_clear_text(self, text, stopwords_file=r"I:\img\model\stopwords_hagongda.txt"):
        import jieba
        with open(stopwords_file, 'r') as f:
            stop_words_str = f.read()
        f_stop_seg_list = stop_words_str.split('\n')
        mywordlist = []
        seg_list = jieba.cut(text, cut_all=False)
        for myword in seg_list:
            if not (myword.strip() in f_stop_seg_list) and len(myword.strip()) > 0:
                mywordlist.append(myword)
        return ' '.join(mywordlist)

    def get_jieba_cut_clear_texts(self, texts):
        clear_texts = []
        for text in texts:
            clear_texts.append(self.get_jieba_cut_clear_text(text))
        return clear_texts

    def get_noun_from_text(self, text):
        import jieba.posseg as pseg
        words = pseg.cut(text)
        nouns = []
        stop_nouns = ["cm", "时", "时期"]
        for word in words:
            if 'n' in word.flag and not word.word in stop_nouns:
                nouns.append(word.word)
        return ' '.join(nouns)

    def get_noun_from_texts(self, texts):
        noun_texts = []
        for text in texts:
            noun_texts.append(self.get_noun_from_text(text))
        return noun_texts

    def get_origin_texts_from_ids(self, ids):
        texts = []
        client = MongoClient()
        db = client.get_database("dpm_all")
        col = db.get_collection("img_id_and_texts")
        recs = list(col.find())
        for img_id in ids:
            texts.append(recs[img_id]["texts"])
        return texts

    def get_jiebacuttext_from_ids(self, ids):
        return self.get_jieba_cut_clear_texts(self.get_origin_texts_from_ids(ids))

    def get_nountext_from_ids(self, ids):
        return self.get_noun_from_texts(self.get_origin_texts_from_ids(ids))

    def get_id2noun_from_db(self):
        col = get_col("dpm_all", "id2noun")
        return list(col.find())


