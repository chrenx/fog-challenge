# Data
**dataset_fog_release**:
- subject 4 and 10 didn't show ant freeze
- data format in column:
    1. time (millisec)
    1. ankle acc - horizontal forward (mg)
    1. ankle acc - vertical (mg)
    1. ankle acc - horizontal lateral (mg)
    1. upper leg acc - horizontal forward (mg)
    1. upper leg acc - vertical (mg)
    1. upper leg acc - horizontal lateral (mg)
    1. trunk acc - horizontal forward (mg)
    1. trunk acc - vertical (mg)
    1. trunk acc - horizontal lateral (mg)
    1. annotations (0,1,2)
        - 0: not part of experiment
        - 1: experiment, no freeze
        - 2: freeze

**gait-in-parkinsons-disease-1.0.0**（可以用来improve当task是walking的时候）  
vertical ground reaction force records of subjects as they walked at their usual, self-selected pace for approximately 2 minutes on level ground
- data format in column:
    - 1: time (secs)  
    - 2-9: VGRF on 8 sensors under left foot
    - 10-17: VGRF on 8 sensors under right foot
    - 18: total force under left foot
    - 19: total force under right foor