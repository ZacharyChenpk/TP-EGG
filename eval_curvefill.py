from sklearn.metrics import precision_recall_curve
from sklearn import metrics
import random
import os
for method in ['TPEGG_LH_reannofix']:
    auc = 0
    auc_fpr_tpr = 0
    for _ in range(50):
        if not os.path.exists('~/gfiles/results/pr_rec/'+method+'_Y.txt'):
            continue
        y = []
        y_pred = []
        with open('gfiles/results/pr_rec/'+method+'_Y.txt', 'r') as f:
            for l in f.readlines():
                ll = l.strip().split()
                y.append(int(ll[0]))
                yp = float(ll[1])
                if yp == 0.:
                    yp = random.random()*1e-6
                if yp == 1.:
                    yp = 1-random.random()*1e-7
                y_pred.append(yp)
        (precisions, recalls, thresholds) = precision_recall_curve(y, y_pred)
        xs = []
        ys = []
        for i, p in enumerate(precisions[:-1]):
            if p >= .5:
                xs.append(recalls[i])
                ys.append(p)

        auc += metrics.auc(xs, ys)

        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
        auc_fpr_tpr += metrics.auc(fpr, tpr)
    print(method,'auc',auc/50)
    print('auc_fpr_tpr',auc_fpr_tpr/50)