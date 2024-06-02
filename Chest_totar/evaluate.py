import numpy as np
import sklearn
import sys
from sklearn import metrics

gs=np.loadtxt('gs.txt')
pred=np.loadtxt('prediction.txt')

ACT=open('activity.list','r')
i=0
act_to_id={}
for line in ACT:
    line=line.strip()
    act_to_id[line]=i
    i=i+1
ACT.close()

all_acts=act_to_id.keys()
OUT=open('auc.'+sys.argv[1],'w')
(a,b)=gs.shape
i=0
for aaa in all_acts:
    fpr, tpr, thresholds =sklearn.metrics.roc_curve(gs[:,i], pred[:,i])
    auc=metrics.auc(fpr, tpr)
    OUT.write(aaa)
    OUT.write('\t')
    OUT.write(str(auc))
    OUT.write('\n')
    i=i+1
