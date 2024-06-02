import sys
import random
import glob

random.seed(sys.argv[1])
all_id=glob.glob('../../data/processed_files/Lower_Back/*_0000000000*')
TRAIN=open('train.id','w')
TEST=open('test.id','w')
for the_id in all_id:
    t=the_id.split('0000000000_')
    rrr=random.random()

    if (rrr<0.8):
        TRAIN.write(t[-1])
        TRAIN.write('\n')
    else:
        TEST.write(t[-1])
        TEST.write('\n')
TRAIN.close()
TEST.close()

all_act=glob.glob('../../data/processed_files/*')
ACT=open('location.list','w')
for act in all_act:
    t=act.split('/')
    ACT.write(t[-1])
    ACT.write('\n')
ACT.close()

all_act=glob.glob('../../data/processed_files/Lower_Back/6011553_0000000000_3-29_2_activities.csv/*')

ACT=open('activity.list','w')
for act in all_act:
    t=act.split('/')
    ACT.write(t[-1])
    ACT.write('\n')
ACT.close()
