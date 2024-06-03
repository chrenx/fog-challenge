
import os
os.system('module load python/3.8.5')

for i in [0,1,2,3,4]:
    os.system('python3 split.py '+str(i))
    REF=open('location.list','r')
    for line in REF:
        line=line.strip()
        os.system('python3 train_lightgbm.py '+line+' '+str(i))

    REF.close()

