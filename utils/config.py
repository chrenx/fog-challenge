ALL_DATASETS = ['dataset_fog_release', 'kaggle_pd_data']

VALIDATION_SET = ["rectified_5_dataset_fog_release", 
                  "rectified_14_dataset_fog_release"]

FEATURES_LIST = ['Annotation',
                 'L_Ankle_Acc_X', 'L_Ankle_Acc_Y', 'L_Ankle_Acc_Z', 
                 'L_MidLatThigh_Acc_X', 'L_MidLatThigh_Acc_Y', 'L_MidLatThigh_Acc_Z',
                 'LowerBack_Acc_X', 'LowerBack_Acc_Y', 'LowerBack_Acc_Z']

PHASE2_DATA_HEAD = [
    'Time',
    # --------------------------------------------------------------------------------
    'GeneralEvent',                   # (t,2)
    'ClinicalEvent',
    # --------------------------------------------------------------------------------
    'L Foot Contact',                 # (t,8)
    'R Foot Contact',
    'L Foot Pressure',
    'R Foot Pressure',
    'Walkway_X',
    'Walkway_Y',
    'WalkwayPressureLevel',           
    'WalkwayFoot',
    # --------------------------------------------------------------------------------
    #                                 13个主要躯干部位
    'LowerBack_Acc_X',                # (t,84)  13*6 + 2*3
    'LowerBack_Acc_Y',
    'LowerBack_Acc_Z',
    'LowerBack_FreeAcc_E',
    'LowerBack_FreeAcc_N',
    'LowerBack_FreeAcc_U',

    'LowerBack_Gyr_X',                # (t,45)  13*3 + 2*3
    'LowerBack_Gyr_Y',
    'LowerBack_Gyr_Z',

    'LowerBack_Mag_X',                # (t,39)  13*3
    'LowerBack_Mag_Y',
    'LowerBack_Mag_Z',

    'LowerBack_VelInc_X',             # (t,39)  13*3
    'LowerBack_VelInc_Y',
    'LowerBack_VelInc_Z',

    'LowerBack_OriInc_q0',            # (t,91)  13*7
    'LowerBack_OriInc_q1',
    'LowerBack_OriInc_q2',
    'LowerBack_OriInc_q3',
    'LowerBack_Roll',
    'LowerBack_Pitch',
    'LowerBack_Yaw',

    'R_Wrist_Acc_X','R_Wrist_Acc_Y','R_Wrist_Acc_Z','R_Wrist_FreeAcc_E','R_Wrist_FreeAcc_N','R_Wrist_FreeAcc_U','R_Wrist_Gyr_X','R_Wrist_Gyr_Y','R_Wrist_Gyr_Z','R_Wrist_Mag_X','R_Wrist_Mag_Y','R_Wrist_Mag_Z','R_Wrist_VelInc_X','R_Wrist_VelInc_Y','R_Wrist_VelInc_Z','R_Wrist_OriInc_q0','R_Wrist_OriInc_q1','R_Wrist_OriInc_q2','R_Wrist_OriInc_q3','R_Wrist_Roll','R_Wrist_Pitch','R_Wrist_Yaw',

    'L_Wrist_Acc_X','L_Wrist_Acc_Y','L_Wrist_Acc_Z','L_Wrist_FreeAcc_E','L_Wrist_FreeAcc_N','L_Wrist_FreeAcc_U','L_Wrist_Gyr_X','L_Wrist_Gyr_Y','L_Wrist_Gyr_Z','L_Wrist_Mag_X','L_Wrist_Mag_Y','L_Wrist_Mag_Z','L_Wrist_VelInc_X','L_Wrist_VelInc_Y','L_Wrist_VelInc_Z','L_Wrist_OriInc_q0','L_Wrist_OriInc_q1','L_Wrist_OriInc_q2','L_Wrist_OriInc_q3','L_Wrist_Roll','L_Wrist_Pitch','L_Wrist_Yaw',

    'R_MidLatThigh_Acc_X','R_MidLatThigh_Acc_Y','R_MidLatThigh_Acc_Z','R_MidLatThigh_FreeAcc_E','R_MidLatThigh_FreeAcc_N','R_MidLatThigh_FreeAcc_U','R_MidLatThigh_Gyr_X','R_MidLatThigh_Gyr_Y','R_MidLatThigh_Gyr_Z','R_MidLatThigh_Mag_X','R_MidLatThigh_Mag_Y','R_MidLatThigh_Mag_Z','R_MidLatThigh_VelInc_X','R_MidLatThigh_VelInc_Y','R_MidLatThigh_VelInc_Z','R_MidLatThigh_OriInc_q0','R_MidLatThigh_OriInc_q1','R_MidLatThigh_OriInc_q2','R_MidLatThigh_OriInc_q3','R_MidLatThigh_Roll','R_MidLatThigh_Pitch','R_MidLatThigh_Yaw',

    'L_MidLatThigh_Acc_X','L_MidLatThigh_Acc_Y','L_MidLatThigh_Acc_Z','L_MidLatThigh_FreeAcc_E','L_MidLatThigh_FreeAcc_N','L_MidLatThigh_FreeAcc_U','L_MidLatThigh_Gyr_X','L_MidLatThigh_Gyr_Y','L_MidLatThigh_Gyr_Z','L_MidLatThigh_Mag_X','L_MidLatThigh_Mag_Y','L_MidLatThigh_Mag_Z','L_MidLatThigh_VelInc_X','L_MidLatThigh_VelInc_Y','L_MidLatThigh_VelInc_Z','L_MidLatThigh_OriInc_q0','L_MidLatThigh_OriInc_q1','L_MidLatThigh_OriInc_q2','L_MidLatThigh_OriInc_q3','L_MidLatThigh_Roll','L_MidLatThigh_Pitch','L_MidLatThigh_Yaw',

    'R_LatShank_Acc_X','R_LatShank_Acc_Y','R_LatShank_Acc_Z','R_LatShank_FreeAcc_E','R_LatShank_FreeAcc_N','R_LatShank_FreeAcc_U','R_LatShank_Gyr_X','R_LatShank_Gyr_Y','R_LatShank_Gyr_Z','R_LatShank_Mag_X','R_LatShank_Mag_Y','R_LatShank_Mag_Z','R_LatShank_VelInc_X','R_LatShank_VelInc_Y','R_LatShank_VelInc_Z','R_LatShank_OriInc_q0','R_LatShank_OriInc_q1','R_LatShank_OriInc_q2','R_LatShank_OriInc_q3','R_LatShank_Roll','R_LatShank_Pitch','R_LatShank_Yaw',

    'L_LatShank_Acc_X','L_LatShank_Acc_Y','L_LatShank_Acc_Z','L_LatShank_FreeAcc_E','L_LatShank_FreeAcc_N','L_LatShank_FreeAcc_U','L_LatShank_Gyr_X','L_LatShank_Gyr_Y','L_LatShank_Gyr_Z','L_LatShank_Mag_X','L_LatShank_Mag_Y','L_LatShank_Mag_Z','L_LatShank_VelInc_X','L_LatShank_VelInc_Y','L_LatShank_VelInc_Z','L_LatShank_OriInc_q0','L_LatShank_OriInc_q1','L_LatShank_OriInc_q2','L_LatShank_OriInc_q3','L_LatShank_Roll','L_LatShank_Pitch','L_LatShank_Yaw',

    'R_DorsalFoot_Acc_X','R_DorsalFoot_Acc_Y','R_DorsalFoot_Acc_Z','R_DorsalFoot_FreeAcc_E','R_DorsalFoot_FreeAcc_N','R_DorsalFoot_FreeAcc_U','R_DorsalFoot_Gyr_X','R_DorsalFoot_Gyr_Y','R_DorsalFoot_Gyr_Z','R_DorsalFoot_Mag_X','R_DorsalFoot_Mag_Y','R_DorsalFoot_Mag_Z','R_DorsalFoot_VelInc_X','R_DorsalFoot_VelInc_Y','R_DorsalFoot_VelInc_Z','R_DorsalFoot_OriInc_q0','R_DorsalFoot_OriInc_q1','R_DorsalFoot_OriInc_q2','R_DorsalFoot_OriInc_q3','R_DorsalFoot_Roll','R_DorsalFoot_Pitch','R_DorsalFoot_Yaw',

    'L_DorsalFoot_Acc_X','L_DorsalFoot_Acc_Y','L_DorsalFoot_Acc_Z','L_DorsalFoot_FreeAcc_E','L_DorsalFoot_FreeAcc_N','L_DorsalFoot_FreeAcc_U','L_DorsalFoot_Gyr_X','L_DorsalFoot_Gyr_Y','L_DorsalFoot_Gyr_Z','L_DorsalFoot_Mag_X','L_DorsalFoot_Mag_Y','L_DorsalFoot_Mag_Z','L_DorsalFoot_VelInc_X','L_DorsalFoot_VelInc_Y','L_DorsalFoot_VelInc_Z','L_DorsalFoot_OriInc_q0','L_DorsalFoot_OriInc_q1','L_DorsalFoot_OriInc_q2','L_DorsalFoot_OriInc_q3','L_DorsalFoot_Roll','L_DorsalFoot_Pitch','L_DorsalFoot_Yaw',

    'R_Ankle_Acc_X','R_Ankle_Acc_Y','R_Ankle_Acc_Z','R_Ankle_FreeAcc_E','R_Ankle_FreeAcc_N','R_Ankle_FreeAcc_U','R_Ankle_Gyr_X','R_Ankle_Gyr_Y','R_Ankle_Gyr_Z','R_Ankle_Mag_X','R_Ankle_Mag_Y','R_Ankle_Mag_Z','R_Ankle_VelInc_X','R_Ankle_VelInc_Y','R_Ankle_VelInc_Z','R_Ankle_OriInc_q0','R_Ankle_OriInc_q1','R_Ankle_OriInc_q2','R_Ankle_OriInc_q3','R_Ankle_Roll','R_Ankle_Pitch','R_Ankle_Yaw',

    'L_Ankle_Acc_X','L_Ankle_Acc_Y','L_Ankle_Acc_Z','L_Ankle_FreeAcc_E','L_Ankle_FreeAcc_N','L_Ankle_FreeAcc_U','L_Ankle_Gyr_X','L_Ankle_Gyr_Y','L_Ankle_Gyr_Z','L_Ankle_Mag_X','L_Ankle_Mag_Y','L_Ankle_Mag_Z','L_Ankle_VelInc_X','L_Ankle_VelInc_Y','L_Ankle_VelInc_Z','L_Ankle_OriInc_q0','L_Ankle_OriInc_q1','L_Ankle_OriInc_q2','L_Ankle_OriInc_q3','L_Ankle_Roll','L_Ankle_Pitch','L_Ankle_Yaw',

    'Xiphoid_Acc_X','Xiphoid_Acc_Y','Xiphoid_Acc_Z','Xiphoid_FreeAcc_E','Xiphoid_FreeAcc_N','Xiphoid_FreeAcc_U','Xiphoid_Gyr_X','Xiphoid_Gyr_Y','Xiphoid_Gyr_Z','Xiphoid_Mag_X','Xiphoid_Mag_Y','Xiphoid_Mag_Z','Xiphoid_VelInc_X','Xiphoid_VelInc_Y','Xiphoid_VelInc_Z','Xiphoid_OriInc_q0','Xiphoid_OriInc_q1','Xiphoid_OriInc_q2','Xiphoid_OriInc_q3','Xiphoid_Roll','Xiphoid_Pitch','Xiphoid_Yaw',

    'Forehead_Acc_X','Forehead_Acc_Y','Forehead_Acc_Z','Forehead_FreeAcc_E','Forehead_FreeAcc_N','Forehead_FreeAcc_U','Forehead_Gyr_X','Forehead_Gyr_Y','Forehead_Gyr_Z','Forehead_Mag_X','Forehead_Mag_Y','Forehead_Mag_Z','Forehead_VelInc_X','Forehead_VelInc_Y','Forehead_VelInc_Z','Forehead_OriInc_q0','Forehead_OriInc_q1','Forehead_OriInc_q2','Forehead_OriInc_q3','Forehead_Roll','Forehead_Pitch','Forehead_Yaw',

    # --------------------------------------------------------------------------------
    # (t,38) 

    'LPressure1','LPressure2','LPressure3','LPressure4','LPressure5','LPressure6','LPressure7','LPressure8','LPressure9','LPressure10','LPressure11','LPressure12','LPressure13','LPressure14','LPressure15','LPressure16',

    'Linsole:Acc_X',
    'Linsole:Acc_Y',
    'Linsole:Acc_Z',

    'Linsole:Gyr_X',
    'Linsole:Gyr_Y',
    'Linsole:Gyr_Z',

    'LTotalForce',

    'LCoP_X',
    'LCoP_Y',

    'RPressure1','RPressure2','RPressure3','RPressure4','RPressure5','RPressure6','RPressure7','RPressure8','RPressure9','RPressure10','RPressure11','RPressure12','RPressure13','RPressure14','RPressure15','RPressure16',

    'Rinsole:Acc_X',
    'Rinsole:Acc_Y',
    'Rinsole:Acc_Z',

    'Rinsole:Gyr_X',
    'Rinsole:Gyr_Y',
    'Rinsole:Gyr_Z',

    'RTotalForce',

    'RCoP_X',
    'RCoP_Y'

]

