import os
import numpy as np
import shutil
from sklearn.svm import LinearSVC
from multiclass.modelUtil import *
from keras.models import load_model
from multiclass.dpmDataset import load
from utils.db_tool import get_col

log_dir = ".\\log"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
tb = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    # embeddings_freq=1,
)


def load_f(name, save_dir=r"I:\img\dpm\dataset\type_enhanced_1p3_split_tvt"):
    import pickle
    import os
    with open(os.path.join(save_dir, "trn_%s" % name), 'rb') as f:
        trn = pickle.load(f)
    with open(os.path.join(save_dir, "tst_%s" % name), 'rb') as f:
        tst = pickle.load(f)
    with open(os.path.join(save_dir, "val_%s" % name), 'rb') as f:
        val = pickle.load(f)
    print("load %s" % name, "done")
    return trn, val, tst


def get_baseline_acc(y):
    """get random guess acc on label list y"""
    acc = []
    for i in range(100):
        cy = y.copy()
        np.random.shuffle(cy)
        acc.append(sum([1 if x else 0 for x in cy==y]) / len(cy))
    return np.mean(acc), np.std(acc)


def svm_clf(name, save_dir=r"I:\img\dpm\dataset\type_enhanced_1p3_split_tvt"):
    trn, val, tst = load_f(name, save_dir=save_dir)

    clf = LinearSVC(dual=False, C=1)
    print("linear svm train...")
    clf.fit(np.reshape(trn["data"], (len(trn["data"]), -1)), trn["y"])
    trn_acc = clf.score(np.reshape(trn["data"], (len(trn["data"]), -1)), trn["y"])
    val_acc = clf.score(np.reshape(val["data"], (len(val["data"]), -1)), val["y"])
    tst_acc = clf.score(np.reshape(tst["data"], (len(tst["data"]), -1)), tst["y"])
    return trn_acc, val_acc, tst_acc


def svm_clf_n(n, name, save_dir=r"I:\img\dpm\dataset\type_enhanced_1p3_split_tvt"):
    trn_accs, val_accs, tst_accs = [], [], []
    for i in range(n):
        x, y, z = svm_clf(name, save_dir=save_dir)
        trn_accs.append(x)
        val_accs.append(y)
        tst_accs.append(z)
    trn = (np.mean(trn_accs), np.std(trn_accs))
    val = (np.mean(val_accs), np.std(val_accs))
    tst = (np.mean(tst_accs), np.std(tst_accs))
    return trn, val, tst


def mlp_clf(name, mlp, save_dir=r"I:\img\dpm\dataset\type_enhanced_1p3_split_tvt", epoch=100, verbose=0, callback=[]):
    rmsprop = keras.optimizers.RMSprop(lr=1e-4)
    mlp.compile(optimizer=rmsprop,
                loss=keras.losses.categorical_crossentropy,
                metrics=['acc'])
    trn, val, tst = load_f(name, save_dir=save_dir)
    mlp.fit(np.reshape(trn["data"], (len(trn["data"]), -1)), keras.utils.to_categorical(trn["y"]),
            batch_size=128, epochs=epoch, verbose=verbose, callbacks=callback,
            validation_data=(np.reshape(val["data"], (len(val["data"]), -1)), keras.utils.to_categorical(val["y"])),
            shuffle=True)
    trn_acc = mlp.evaluate(np.reshape(trn["data"], (len(trn["data"]), -1)), keras.utils.to_categorical(trn["y"]), verbose=0)
    val_acc = mlp.evaluate(np.reshape(val["data"], (len(val["data"]), -1)), keras.utils.to_categorical(val["y"]), verbose=0)
    tst_acc = mlp.evaluate(np.reshape(tst["data"], (len(tst["data"]), -1)), keras.utils.to_categorical(tst["y"]), verbose=0)
    mlp.save(os.path.join(".\\model", name))
    return trn_acc, val_acc, tst_acc


def mlp_clf_n(n, name, mlp, save_dir=r"I:\img\dpm\dataset\type_enhanced_1p3_split_tvt", epoch=50, verbose=0, callback=[]):
    """
    return means and std for acc of trn/val/tst
    """
    trn_accs, val_accs, tst_accs = [], [], []
    for i in range(n):
        x, y, z = mlp_clf(name, mlp, save_dir=save_dir, epoch=epoch, verbose=verbose, callback=callback)
        trn_accs.append(x[1])
        val_accs.append(y[1])
        tst_accs.append(z[1])
    trn = (np.mean(trn_accs), np.std(trn_accs))
    val = (np.mean(val_accs), np.std(val_accs))
    tst = (np.mean(tst_accs), np.std(tst_accs))
    return trn, val, tst


def svm_test():
    accs = []
    test_features1 = ["desc_1000_100", "hu", "lbp", "hsv", "hog", "resnet50_ft"]
    for f in test_features1:
        print("-" * 200)
        acc = svm_clf_n(3, f)
        print(acc)
        accs.append(acc)
    return accs


def mlp_test():
    accs = []
    mlps = [
        get_hu_mlp(),
        get_lbp_mlp(),
        get_hsv_mlp(),
        get_hog_mlp(),
        get_resnet50_mlp(),
        get_text_mlp(),
        get_text_lstm(1000, 16, 100),
        get_text_embeding(1000, 16, 100)
    ]
    features = ["hu", "lbp", "hsv", "hog", "resnet50_ft", "desc_1000_100", "desc_1000_100", "desc_1000_100"]
    choice = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in choice:
        print("-" * 200)
        res = mlp_clf(features[i], mlps[i], epoch=300, verbose=0)
        print(res)
        accs.append((features[i], res))
    return accs


def compose_test():
    # composed : train from pretrained
    hu_mlp = load_model(os.path.join(".\\model", "hu"))
    lbp_mlp = load_model(os.path.join(".\\model", "lbp"))
    hsv_mlp = load_model(os.path.join(".\\model", "hsv"))
    hog_mlp = load_model(os.path.join(".\\model", "hog"))
    resnet_mlp = load_model(os.path.join(".\\model", "resnet50_ft"))
    text_mlp = load_model(os.path.join(".\\model", "embedding_16"))

    composed_model = get_composed_model([
        hu_mlp,
        lbp_mlp,
        hsv_mlp,
        hog_mlp,
        resnet_mlp,
        text_mlp
    ])
    composed_model.summary()

    # load data
    trn_y, val_y, tst_y = [], [], []
    trn_x, val_x, tst_x = [], [], []
    for f in [
        "hu",
        "lbp",
        "hsv",
        "hog",
        "resnet50_ft",
        "desc_1000_100",
    ]:
        x, y, z = load(os.path.join(f))
        trn_y.append(x["y"])
        val_y.append(y["y"])
        tst_y.append(z["y"])
        trn_x.append(x["data"])
        val_x.append(y["data"])
        tst_x.append(z["data"])
    trn_y = np.asarray(trn_y)
    val_y = np.asarray(val_y)
    tst_y = np.asarray(tst_y)
    assert (np.sum(np.abs(trn_y - trn_y[0])) == 0)

    composed_model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),
                           loss=keras.losses.categorical_crossentropy,
                           metrics=['acc'])
    composed_model.fit([x for x in trn_x], keras.utils.to_categorical(trn_y[0]),
                       batch_size=64, epochs=100, verbose=1, shuffle=True,
                       validation_data=([x for x in val_x], keras.utils.to_categorical(val_y[0])))
    scores = composed_model.evaluate([x for x in tst_x], keras.utils.to_categorical(tst_y[0]))
    return scores


if __name__ == '__main__':
    col = get_col("keras", "test")

    # accs = svm_test()
    # print(accs)
    # col.insert_one({"svm_mean_std": accs})
    #
    # accs = []
    # for i in range(3):
    #     acc = mlp_test()
    #     accs.append(acc)
    # print(accs)
    # col.insert_one({"mlp_mean_std": accs})

    accs = []
    for i in range(3):
        acc = compose_test()
        accs.append(acc)
    print(accs)
    col.insert_one({"composed_mean_std": accs})

    pass