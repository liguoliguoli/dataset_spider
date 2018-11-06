from pymongo import MongoClient

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import numpy as np


def get_mAP_acc(pred, true):
    """
    return mean and std
    :param pred: 1d array
    :param true: 1d array
    :return:
    """
    accs = []
    for y in range(max(True)):
        t = sum([1 if true[i] == y else 0 for i in range(len(true))])
        tp = sum([1 if pred[i] == y and true[i] == y else 0 for i in range(len(true))])
        accs.append(tp / t)
    return np.mean(accs), np.std(accs)


def get_acc(pred, true):
    """
    return acc
    :param pred: 1d array
    :param true: 1d array
    :return:
    """
    tp = sum([1 if true[i] == pred[i] else 0 for i in range(len(true))])
    return tp / len(true)


def get_mins(x):
    sp = x.split(":")
    return 60 * int(sp[0]) + int(sp[1]) + int(sp[2].split(".")[0]) / 60


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points



client = MongoClient()

db = client.get_database("keras")

col = db.get_collection("tf_compare_res")

res = list(col.find())

res = res[17:]

print("len res", len(res))

rmsprop = [x for x in res if "test3" in x["comment"]]

print("len rmsprop:", len(rmsprop))

sgd = [x for x in res if "test4" in x["comment"]]

print("len sgd:", len(sgd))

model_names = ["vgg16", "vgg19", "inceptionv3", "xception", "resnet50"]

model_res = {}

for x in sgd:
    model_name = x["model_name"]
    #     print(model_name)
    if model_name not in model_res:
        model_res[model_name] = []
    model_res[model_name].append(x)

print(model_res.keys())

print("len(model_res[vgg16])", len(model_res["vgg16"]))

ret = []

for model, res in model_res.items():
    tl_scores = [x["scores"][0][1] for x in res]
    ft_scores = [x["scores"][1][1] for x in res]
    tl_times = [x["times"][0] for x in res]
    tl_times = [get_mins(t) for t in tl_times]
    ft_times = [x["times"][1] for x in res]
    ft_times = [get_mins(t) for t in ft_times]
    ret.append((model, tl_scores, ft_scores, tl_times, ft_times))
print("ret:")
print(ret)

sta = []

for x in ret:
    model = x[0]
    tl_scores = x[1]
    ft_scores = x[2]
    # tl_times = [get_mins(t) for t in x[3]]
    # ft_times = [get_mins(t) for t in x[4]]
    tl_scores = np.asarray(tl_scores)
    ft_scores = np.asarray(ft_scores)
    tl_times = np.asarray(tl_times)
    ft_times = np.asarray(ft_times)
    tl_sc = (np.mean(tl_scores), np.std(tl_scores))
    ft_sc = (np.mean(ft_scores), np.std(ft_scores))
    tl_tm = (np.mean(tl_times), np.std(tl_times))
    ft_tm = (np.mean(ft_times), np.std(ft_times))
    sta.append((model, tl_sc, ft_sc, tl_tm, ft_tm))
print("sta:")
print(sta)


tl_sc = [{x[0]: x[1]} for x in ret]
print("tl_sc:")
print(tl_sc)

plt.figure(figsize=(20, 15))

plt.subplot(2, 2, 1, title="tl_val_acc")
colors = ["b", "g", "r", "m", "y"]
i = 0
for model, res in model_res.items():
    hists = res[2]["hists"]
    tl_hist = hists[0]
    #     smoothed_acc = smooth_curve(tl_hist["acc"])
    #     print(smoothed_acc)
    #     plt.plot(range(40),smooth_curve(tl_hist["acc"])[10:],label="acc")
    plt.plot(range(40), smooth_curve(tl_hist["val_acc"])[10:], colors[i], label=model)
    plt.xticks(range(0, 40, 5), range(10, 50, 5))
    plt.legend()
    plt.title("tl_val_acc")

    i += 1

plt.subplot(2, 2, 2, title="ft_val_acc")
colors = ["b", "g", "r", "m", "y"]
i = 0
for model, res in model_res.items():
    hists = res[2]["hists"]
    ft_hist = hists[1]
    #     smoothed_acc = smooth_curve(tl_hist["acc"])
    #     print(smoothed_acc)
    #     plt.plot(range(40),smooth_curve(tl_hist["acc"])[10:],label="acc")
    plt.plot(range(40), smooth_curve(ft_hist["val_acc"])[10:], colors[i], label=model)
    plt.xticks(range(0, 40, 5), range(10, 50, 5))
    plt.legend()
    plt.title("ft_val_acc")

    i += 1

plt.subplot(2, 2, 3, title="tl_val_loss")
colors = ["b", "g", "r", "m", "y"]
i = 0
for model, res in model_res.items():
    hists = res[2]["hists"]
    tl_hist = hists[0]
    #     smoothed_acc = smooth_curve(tl_hist["acc"])
    #     print(smoothed_acc)
    #     plt.plot(range(40),smooth_curve(tl_hist["acc"])[10:],label="acc")
    plt.plot(range(40), smooth_curve(tl_hist["val_loss"])[10:], colors[i], label=model)
    plt.xticks(range(0, 40, 5), range(10, 50, 5))
    plt.legend()
    plt.title("tl_val_loss")

    i += 1

plt.subplot(2, 2, 4, title="ft_val_loss")
colors = ["b", "g", "r", "m", "y"]
i = 0
for model, res in model_res.items():
    hists = res[2]["hists"]
    ft_hist = hists[1]
    #     smoothed_acc = smooth_curve(tl_hist["acc"])
    #     print(smoothed_acc)
    #     plt.plot(range(40),smooth_curve(tl_hist["acc"])[10:],label="acc")
    plt.plot(range(40), smooth_curve(ft_hist["val_loss"])[10:], colors[i], label=model)
    plt.xticks(range(0, 40, 5), range(10, 50, 5))
    plt.legend()
    plt.title("ft_val_loss")

    i += 1
plt.show()