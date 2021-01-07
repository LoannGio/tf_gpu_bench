import glob
import os
import json
from matplotlib import pyplot as plt

def sortMultipleListsFromOne(target, others, reverse=False):
    for i in range(len(others)):
        assert len(target) == len(others[i])
    
    target_dic = {k:v for k,v in enumerate(target)}
    others_dic = []
    for o in others:
        others_dic.append({k:v for k,v in enumerate(o)})
        
    sorted_target = {k: v for k, v in sorted(target_dic.items(), key=lambda item: item[1], reverse=reverse)}
    
    others_res = []
    for o_d in others_dic:
        o = []
        for k in sorted_target.keys():
            o.append(o_d[k])
        others_res.append(o)
    return list(sorted_target.values()), others_res


def displayResults(res_path, plot_path):
    files = glob.glob(os.path.join(res_path, "*.json"))
    X = []
    Y1 = []
    Y2 = []
    for file in files:
        with open(file, "r") as f:
            js_data = json.load(f)
            x = js_data["GPU"]+ " \n " + js_data["host"]
            y1 = float(js_data["performance_time"].split(" ")[0])
            y2 = js_data["best_acc"]
            X.append(x)
            Y1.append(y1)
            Y2.append(y2)
    plt.rcParams["figure.figsize"] = (10,10)
    fig, ax1 = plt.subplots()
    
    Y1, [X, Y2] = sortMultipleListsFromOne(Y1, [X, Y2])
    #Plot time
    ax1.bar(x=X, height=Y1)
    ax1.set_xticklabels(labels=X,rotation=45, ha="right")
    ax1.set_ylabel("Runtime (seconds)", color="C0")
    ax1.set_xlabel("GPU / host")
    ax1.tick_params(axis='y', labelcolor="C0")


    #Plot acc
    ax2 = ax1.twinx()
    ax2.scatter(x=X,y=Y2, s=[200 for _ in range(len(X))], label="best accuracy", c=["C1"])
    ax2.set_ylabel("Best Accuracy", color="C1")
    #ax2.set_ylim(0,1)
    ax2.tick_params(axis='y', labelcolor="C1")


    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "comparison.pdf"))
    plt.savefig(os.path.join(plot_path, "comparison.png"))

displayResults("./res", "./plots")