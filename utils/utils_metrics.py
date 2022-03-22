import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def evaluteTop1_5(classfication, lines, metrics_out_path):
    correct_1 = 0
    correct_5 = 0
    preds   = []
    labels  = []
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred        = classfication.detect_image(x)
        pred_1      = np.argmax(pred)
        correct_1   += pred_1 == y
        
        pred_5      = np.argsort(pred)[::-1]
        pred_5      = pred_5[:5]
        correct_5   += y in pred_5
        
        preds.append(pred_1)
        labels.append(y)
        if index % 100 == 0:
            print("[%d/%d]"%(index, total))
            
    hist        = fast_hist(np.array(labels), np.array(preds), len(classfication.class_names))
    Recall      = per_class_Recall(hist)
    Precision   = per_class_Precision(hist)
    
    show_results(metrics_out_path, hist, Recall, Precision, classfication.class_names)
    return correct_1 / total, correct_5 / total, Recall, Precision

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()
    
def show_results(miou_out_path, hist, Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            
def evaluteRecall(classfication, lines, metrics_out_path):
    correct = 0
    total = len(lines)
    
    preds   = []
    labels  = []
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        pred = np.argmax(pred)
        
        preds.append(pred)
        labels.append(y)
        
    hist        = fast_hist(labels, preds, len(classfication.class_names))
    Recall      = per_class_Recall(hist)
    Precision   = per_class_Precision(hist)
    
    show_results(metrics_out_path, hist, Recall, Precision, classfication.class_names)
    return correct / total
