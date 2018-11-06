import keras
from keras.applications import VGG16,VGG19,ResNet50,InceptionV3,Xception
from keras.layers import Dropout,Dense,Activation,BatchNormalization,Embedding,Flatten
from keras.models import Model,Sequential
from keras.optimizers import RMSprop,SGD
from keras.losses import categorical_crossentropy
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,NumpyArrayIterator
import os
from imutils import paths
from datetime import datetime
from pymongo import MongoClient
import shutil


class TransferLearner(object):

    def __init__(self, cnn_model, data_dir, dst_dir,  model_name, mark, target_size=(224, 224)):
        """
        for convenience , for transfer learning experiment
        :param cnn_model: transfer learning used cnn model, like VGG16, ResNet50, and should be 'included_top=False'
        :param data_dir: data for transfer learning
        :param dst_dir: dir for save trained model and  plot images
        :param model_name: model type , one of ["vgg16", "inceptionV3", ...]
        :param mark: mark to different test on same model, like "vgg16_1", "vgg16_2"
        :param target_size: target size for resizing the input images
        """
        self.cnn_model = cnn_model
        self.model_name = model_name
        self.mark = mark
        self.data_dir = data_dir
        self.save_dir = os.path.join(dst_dir, self.mark)
        self.composed_model = None
        self.transfer_learned = False
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.val_dir = os.path.join(self.data_dir, "val")
        self.train_imgs = list(paths.list_images(self.train_dir))
        self.test_imgs = list(paths.list_images(self.test_dir))
        self.val_imgs = list(paths.list_images(self.val_dir))
        self.hists = []
        self.target_size = target_size
        self.scores = []
        self.times = []
        self.nb = len(list(os.listdir(self.train_dir)))
        self.epoches = []
        self.lrs = []
        self.freezing_layer_index = None
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print("create dir ", self.save_dir)
        print("load model", model_name,
              "target_size", target_size,
              "datadir", data_dir,
              "save_dir", self.save_dir)
        self.set_composed_model()

    def set_composed_model(self, dim=256, atv="relu"):
        x = self.cnn_model.output
        x = Flatten()(x)
        x = Dense(dim)(x)
        x = BatchNormalization()(x)
        x = Activation(activation=atv)(x)
        x = Dense(self.nb)(x)
        x = Activation("softmax")(x)
        model = Model(inputs=self.cnn_model.input, outputs=x)
        self.composed_model = model
        # self.composed_model.summary()

    def plot_hist(self, hist, title):
        """

        :param hist: should be a dict contained "acc","val_acc","loss","val_loss"
        :param title:
        :return:
        """
        acc = hist['acc']
        val_acc = hist['val_acc']
        loss = hist['loss']
        val_loss = hist['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        # plt.ylim(0, 1)
        plt.title('Training and validation accuracy [%s]'%title)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss [%s]'%title)
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "%s_%s.jpg"%(self.mark, title)))
        # plt.show()

    def freezing_layers(self, freezing_layer_index):
        """

        :param freezing_layer_index: index of the layer you want to start trainable
            [0 : freezing_layer_index] is not trainable,
            [freezing_layer_index :] is trainable
        :return:
        """
        for layer in self.cnn_model.layers:
            layer.trainable = False
        if freezing_layer_index == -1:
            return
        for layer in self.cnn_model.layers[freezing_layer_index:]:
            layer.trainable = True
        self.freezing_layer_index = freezing_layer_index

    def print_layer_trainable(self):
        """
        just for check
        :return:
        """
        for i, layer in enumerate(self.cnn_model.layers):
            print(i, layer.name, layer.trainable)

    def get_index_of_layer(self, layer_name):
        """
        just for check
        :param layer_name:
        :return:
        """
        layer_names = [layer.name for layer in self.cnn_model.layers]
        idx = -1
        if layer_name in layer_names:
            idx = layer_names.index(layer_name)
        return idx, layer_names

    def transfer_learning(self, epoch=50, batch_size=32, lr=5*1e-4, verbose=True):
        """
        set freezing_layer_index to -1,
        set data generator for train, val,
        train
        record time, score, epoch, lr, and plot images
        :param epoch:
        :param batch_size:
        :param lr:
        :param verbose:
        :return:
        """
        print("BEGIN transfer learning %s ......"% self.mark)
        print("-"*150)
        self.freezing_layers(-1)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical')
        self.composed_model.compile(
            optimizer=RMSprop(lr=lr),
            loss=categorical_crossentropy,
            metrics=['acc', 'mae'])
        self.print_layer_trainable()
        t = datetime.now()
        his1 = self.composed_model.fit_generator(
            train_generator,
            epochs=epoch,
            steps_per_epoch=len(self.train_imgs) // batch_size,
            validation_data=validation_generator,
            validation_steps=len(self.val_imgs) // batch_size,
            verbose=verbose
        )
        t = datetime.now() - t
        self.times.append(str(t))
        print(t)
        self.hists.append(his1.history)
        self.get_score(batch_size)
        self.epoches.append(epoch)
        self.lrs.append(lr)
        self.plot_hist(his1.history, "transfer learning train hist")

    def get_score(self, batch_size):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical')
        score = self.composed_model.evaluate_generator(
            test_generator,
            steps=len(self.test_imgs) // batch_size
        )
        print(score)
        self.scores.append(score)

    def fine_tuning(self, freezing_layer_index, lr=5*1e-4, epoch=50, batch_size=32, verbose=True):
        print("BEGIN fine tuning %s ......" % self.mark)
        print("-" * 150)
        self.freezing_layers(freezing_layer_index)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical')
        self.composed_model.compile(
            optimizer=SGD(lr=lr, momentum=0.9),
            loss=categorical_crossentropy,
            metrics=['acc', 'mae'])
        self.print_layer_trainable()
        t = datetime.now()
        his2 = self.composed_model.fit_generator(
            train_generator,
            epochs=epoch,
            steps_per_epoch=len(self.train_imgs) // batch_size,
            validation_data=validation_generator,
            validation_steps=len(self.val_imgs) // batch_size,
            verbose=verbose
        )
        t = datetime.now() - t
        self.times.append(str(t))
        print(t)
        self.hists.append(his2.history)
        self.get_score(batch_size)
        self.epoches.append(epoch)
        self.lrs.append(lr)
        self.plot_hist(his2.history, "fine tuning train hist")

    def save(self, comment=""):
        model_path = os.path.join(self.save_dir, self.mark)
        client = MongoClient()
        db = client.get_database("keras")
        col = db.get_collection("tf_compare_res")
        col.insert(
            {
                "data_dir": self.data_dir,
                "model_name": self.model_name,
                "times": self.times,
                "scores": self.scores,
                "hists": self.hists,
                "model_path": model_path,
                "comment": comment,
                "mark": self.mark,
                "epoches": self.epoches,
                "learning_rates": self.lrs,
                "freezing_layer_index": self.freezing_layer_index
            }
        )

        self.composed_model.save(model_path)


if __name__ == '__main__':
    target_size = 224
    data_dir = r"I:\img\dpm\origin\type_enhanced_1p3_split_tvt"
    test_dir = r"I:\img\test_dir"
    dst_dir = r"I:\img\model"
    epoch = 100
    models = {
        "vgg16": VGG16(weights="imagenet", include_top=False,input_shape=(target_size, target_size, 3)),
        "vgg19": VGG19(weights="imagenet", include_top=False, input_shape=(target_size, target_size, 3)),
        "inceptionv3": InceptionV3(weights="imagenet", include_top=False, input_shape=(target_size, target_size, 3)),
        "xception": Xception(weights="imagenet", include_top=False, input_shape=(target_size, target_size, 3)),
        "resnet50": ResNet50(weights="imagenet", include_top=False, input_shape=(target_size, target_size, 3)),
    }
    freezing_layers = {
        "vgg16": ("block5_conv1", 15, 16),
        "vgg19": ("block5_conv1", 17, 16),
        "inceptionv3": ("conv2d_65", 197, 32),
        "xception": ("add_10", 105, 32),
        "resnet50": ("add_8", 89, 32),
    }
    for i in range(3):
        for n, m in models.items():
            try:
                # if not n in ["inceptionv3"]:
                #     print("pass")
                #     continue
                print("*" * 150)
                print("START", n)
                print("*" * 150)
                # print(m.summary())
                batch_size = freezing_layers[n][2]
                tl = TransferLearner(m, data_dir, os.path.join(dst_dir, "epoch100_sgd", str(i+1)), n, "%s_%s" % (n, i))
                tl.transfer_learning(epoch=epoch, verbose=2, batch_size=batch_size)
                tl.fine_tuning(freezing_layers[n][1], epoch=epoch, verbose=2, batch_size=batch_size)
                tl.save(comment="epoch100_sgd %s"%i)
                del m
            except Exception as e:
                print(n, e)