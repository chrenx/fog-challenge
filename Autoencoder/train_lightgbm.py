import sys
import numpy as np
import glob
import lightgbm as lgb




FILE=open('activity.list','r')
class_map={}
class_num=0

for line in FILE:
    line=line.strip()
    class_map[line]=class_num
    class_num=class_num+1
FILE.close()

params_k = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class':class_num,
           # 'subsample': 0.5,
           # 'subsample_freq': 1,
            'learning_rate': 0.03,
           # 'num_leaves': 2**11-1,
           # 'min_data_in_leaf': 2**12-1,
           # 'feature_fraction': 0.5,
           # 'max_bin': 100,
          #  'n_estimators': 500,
            'n_estimators': 500,
            'verbose': 0,
            'boost_from_average': False,
            "random_seed":42}

location=sys.argv[1]

train_data=[]
train_gs=[]
REF=open('train.id','r')
for the_id in REF:
    the_id=the_id.strip()
    all_file=glob.glob('feature/'+location+'/*'+the_id+'/*')
    for the_file in all_file:
        data=np.load(the_file)
        train_data.append(data)
        gs=class_map[the_file.split('/')[-1]]
        train_gs.append(gs)
REF.close()


train_data = lgb.Dataset(data=np.asarray(train_data),
                         label=np.asarray(train_gs))

model_gbm = lgb.train(params_k, train_data,
              # num_boost_round=2000)
               num_boost_round=200)



test_data=[]
test_gs=[]
REF=open('test.id','r')
for the_id in REF:
    the_id=the_id.strip()
    all_file=glob.glob('feature/'+location+'/*'+the_id+'/*')
    for the_file in all_file:
        data=np.load(the_file)
        test_data.append(data)
        gs=class_map[the_file.split('/')[-1]]
        test_gs.append(gs)
REF.close()

pred=model_gbm.predict(np.asarray(test_data))

print(pred.shape) # 168, 21
np.savetxt(location+'.fold'+sys.argv[2]+'.pred',pred)
np.savetxt(location+'.fold'+sys.argv[2]+'.gs',np.asarray(test_gs))


