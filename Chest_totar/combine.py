import glob
import numpy as np

all_pred=glob.glob('pred.model.*')
val_matrix=np.loadtxt(all_pred[0])
i=1
while (i<len(all_pred)):
    val_matrix=val_matrix+np.loadtxt(all_pred[0])
    i=i+1
val_matrix=val_matrix/i
np.savetxt('prediction.txt', val_matrix,delimiter='\t')

