
# coding: utf-8


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

from matplotlib import pyplot as plt
from img_precess.featureExtracter import FeatureExtractor


class ImagePrecessor(object):

    def __init__(self):
        self.extractor = FeatureExtractor()

    def crop_rectangle(self, f_input, f_out, threshhold, plot=False):
        """
        crop ROI rectangle from f_input, write to f_out
        :param f_input:
        :param f_out:
        :param threshhold:
        :param plot:
        :return:
        """

        image = cv2.imread(f_input, cv2.IMREAD_COLOR)
        # print(image.size)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # 去噪声和二值化
        # blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, threshhold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=12)

        _, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        x, y, w, h = cv2.boundingRect(c)

        margin = 20
        x1 = max(x - margin, 0)
        x2 = min(x + w + margin, image.shape[1])
        y1 = max(y - margin, 0)
        y2 = min(y + h + margin, image.shape[0])
        cropImg = image[y1:y2, x1:x2]

        if plot:
            plt.figure(figsize=(12, 16))
            titles = ["origin", "gradient", "closed", "croped"]
            imgs = [image, gradient, closed, cv2.cvtColor(cropImg, cv2.COLOR_RGB2BGR)]
            for i, img in enumerate(imgs):
                plt.subplot(2, 2, i + 1)
                plt.imshow(img)
                plt.title(titles[i])
                plt.axis("off")
            plt.show()

        cv2.imwrite(f_out, cropImg)

    def crop(self, src_dir, dst_dir, threshhold=60):
        """

        :param src_dir:
        :param dst_dir:
        :return:
        """
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.mkdir(dst_dir)
        id2t = self.extractor.get_id2t_from_db()
        img_paths = paths.list_images(src_dir)
        t1 = datetime.now()
        print("crop", src_dir, "to", dst_dir)
        for i, path in enumerate(img_paths):
            id = path.split("\\")[-1].split(".")[0]
            t = id2t[int(id)]
            sub_dir = os.path.join(dst_dir, t)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            f_dst = os.path.join(sub_dir, "%s.jpg" % id)
            try:
                self.crop_rectangle(path, f_dst, threshhold)
                print(i, f_dst)
            except Exception as e:
                print(path, e)

        print(datetime.now() - t1)


    def remove_noise(self, f_input, f_out, flag="M", plot=False):
        """
        blur
        :param f_input:
        :param f_out:
        :param flag:
        :param plot:
        :return:
        """
        im = cv2.imread(f_input)
        if flag == "M":
            im_blur = cv2.medianBlur(im, 3)
        elif flag == "G":
            im_blur = cv2.GaussianBlur(im, (3, 3), 1.1)
        else:
            im_blur = cv2.blur(im, (3, 3))
        if plot:
            plt.figure(figsize=(20, 15))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(im_blur, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
        cv2.imwrite(f_out, im_blur)

    def blur(self, src_dir, dst_dir, flag="M"):
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.mkdir(dst_dir)
        id2t = self.extractor.get_id2t_from_db()
        img_paths = paths.list_images(src_dir)
        t1 = datetime.now()
        print("crop", src_dir, "to", dst_dir)
        for i, path in enumerate(img_paths):
            id = path.split("\\")[-1].split(".")[0]
            t = id2t[int(id)]
            sub_dir = os.path.join(dst_dir, t)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            f_dst = os.path.join(sub_dir, "%s.jpg" % id)
            try:
                self.remove_noise(path, f_dst, flag)
                print(i, f_dst)
            except Exception as e:
                print(path, e)

        print(datetime.now() - t1)

    def enhance_contrast(self, f_input, f_out, factor, plot=False):
        """
        enhance img contrast from f_input by factor and write enchaced img to f_out
        :param f_input:
        :param f_out:
        :param factor:
        :param plot:
        :return:
        """
        im = Image.open(f_input)
        im_enhanced = ImageEnhance.Contrast(im).enhance(factor)
        if plot:
            plt.figure(figsize=(20, 15))
            plt.subplot(1, 2, 1)
            plt.imshow(im)
            plt.subplot(1, 2, 2)
            plt.imshow(im_enhanced)
            plt.axis("off")
            plt.show()
        # cv2.imwrite(f_out, cv2.cvtColor(im_enhanced, cv2.COLOR_BGR2RGB))
        im_enhanced.save(f_out)

    def enhance(self, src_dir, dst_dir, factor=1.3):
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.mkdir(dst_dir)
        id2t = self.extractor.get_id2t_from_db()
        img_paths = paths.list_images(src_dir)
        t1 = datetime.now()
        print("crop", src_dir, "to", dst_dir)
        for i, path in enumerate(img_paths):
            id = path.split("\\")[-1].split(".")[0]
            t = id2t[int(id)]
            sub_dir = os.path.join(dst_dir, t)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            f_dst = os.path.join(sub_dir, "%s.jpg" % id)
            try:
                self.enhance_contrast(path, f_dst, factor)
                print(i, f_dst)
            except Exception as e:
                print(path, e)

        print(datetime.now() - t1)

    def create_subdir(self, src_dir, dst_dir, img_sl):
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        for x in img_sl:
            id = x["img_id"]
            t = x["type"][0]
            subdir = os.path.join(dst_dir, t)
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            f_src = os.path.join(src_dir, "%s.jpg" % id)
            f_dst = os.path.join(subdir, "%s.jpg" % id)
            print("copy ", f_src, " to ", f_dst)
            try:
                shutil.copy(f_src, f_dst)
            except Exception as e:
                print(f_src, e)

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def create_train_val_test(self, src_dir, dst_dir):
        """
        random shuffle all img in src_dir and split into train\val\test dir
        :param src_dir: should be like base_dir\class_name\img_id.jpg
        :param dst_dir: should be like base_dir\[train|val|test]\class_name\img_id.jpg
        :return: none
        """

        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        imgs = []
        for cls in os.listdir(src_dir):
            cls_ims = os.listdir(os.path.join(src_dir, cls))
            imgs.extend([(cls, im) for im in cls_ims])
        print(len(imgs))

        np.random.shuffle(imgs)

        self.ensure_dir(os.path.join(dst_dir, "train"))
        for i, im in enumerate(imgs[:4000]):
            self.ensure_dir(os.path.join(dst_dir, "train", im[0]))
            f_src = os.path.join(src_dir, im[0], im[1])
            f_dst = os.path.join(dst_dir, "train", im[0], im[1])
            try:
                shutil.copy(f_src, f_dst)
                print(i, f_dst)
            except Exception as e:
                print(f_src, e)

        self.ensure_dir(os.path.join(dst_dir, "val"))
        for i, im in enumerate(imgs[4000:5000]):
            self.ensure_dir(os.path.join(dst_dir, "val", im[0]))
            f_src = os.path.join(src_dir, im[0], im[1])
            f_dst = os.path.join(dst_dir, "val", im[0], im[1])
            try:
                shutil.copy(f_src, f_dst)
                print(i, f_dst)
            except Exception as e:
                print(f_src, e)

        self.ensure_dir(os.path.join(dst_dir, "test"))
        for i, im in enumerate(imgs[5000:]):
            self.ensure_dir(os.path.join(dst_dir, "test", im[0]))
            f_src = os.path.join(src_dir, im[0], im[1])
            f_dst = os.path.join(dst_dir, "test", im[0], im[1])
            try:
                shutil.copy(f_src, f_dst)
                print(i, f_dst)
            except Exception as e:
                print(f_src, e)


if __name__ == '__main__':
    ip = ImagePrecessor()
    ip.create_train_val_test(r"I:\img\dpm\origin\type_enhanced_1p3_split", r"I:\img\dpm\origin\type_enhanced_1p3_split_tvt")










