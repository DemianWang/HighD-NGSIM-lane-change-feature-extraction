#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:15:22 2021

@author: demian Wang ## MIS: wangqi123
"""
import numpy as np
import matplotlib.pyplot as plt
import math

#Basic information of dataset (global information)
DATASET="dataset"
SCENE="scene"
SCENE_AVERAGE_SPEED="scene_avg_speed"

#Local environment information of target (ego) vehicle
CURRENT_LANE_WIDTH="current_lane_width" #in Frenet Coordinate
TOTAL_NUM_LANE="total_num_lane"
ROAD_SPEED_LIMIT="road_speed_limit"
CURRENT_ROAD_CURVATURE="current_road_curvature"
TRAFFIC_LIGHT = "traffic_light"

# Driving maneuver classification
CURRENT_DRIVING_MANEUVER="current_maneuver"
LEFT_RIGHT_MANEUVER="left_right_maveuver"
CURRENT_LANE_FEATURE_ID="current_feature_lane_id"
CURRENT_LANE_AVERAGE_SPEED="current_lane_average_speed"
CURRENT_LANE_SPEED_LIMIT="current_lane_speed_limit"
INTENTION_LANE="intention_lane"

# Target vehicle information
DISTANCE_TO_LANE="Distance_to_lane" 
CLASS = "class"
VEHICLE_WIDTH = "vehicle_width"
VEHICLE_HEIGHT = "vehicle_height"
S_LOCATION = "s_Location"
D_LOCATION = "d_Location"
S_VELOCITY = "s_Velocity"
D_VELOCITY = "d_Velocity"
HEADING_ANGLE = "heading_angle"
S_ACCELERATION = "s_Acceleration"
D_ACCELERATION = "d_Acceleration"
YAW_RATE = "yaw_rate"
S_JERK = "s_jerk"
D_JERK = "d_jerk"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
LEFT_LANE_TYPE="left_lane_type"
RIGHT_LANE_TYPE="right_lane_type"
STEERING_RATE_ENTROPY="steering_rate_entropy"

#Interactive information of surrounding vehicles
IS_PRECEDING_VEHICLE="is_preceding_vehicle"
IS_FOLLOWING_VEHICLE="is_following_vehicle"
IS_LANE_TO_CHANGE_PRECEDING_VEHICLE="is_LTC_preceding_vehicle"
IS_LANE_TO_CHANGE_ALONGSIDE_VEHICLE="is_LTC_alongside_vehicle"
IS_LANE_TO_CHANGE_FOLLOWING_VEHICLE="is_LTC_following_vehicle"
PRECEDING_VEHICLE_FEATURE="preceding_vehicle_feature"
FOLLOWING_VEHICLE_FEATURE="following_vehicle_feature"
LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE="LTC_preceding_vehicle_feature"
LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE="LTC_following_vehicle_feature"
LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE="LTC_alongside_vehicle_feature"
GAP="gap"
FORWARD_LANE_TO_CHANGE_THW="forward_LTC_thw"
FORWARD_LANE_TO_CHANGE_DHW="forward_LTC_dhw"
FORWARD_LANE_TO_CHANGE_TTC="forward_LTC_ttc"
BACKWARD_LANE_TO_CHANGE_THW="backward_LTC_thw"
BACKWARD_LANE_TO_CHANGE_DHW="backward_LTC_dhw"
BACKWARD_LANE_TO_CHANGE_TTC="backward_LTC_ttc"

# TRACK FILE
BBOX = "bbox"
VEHICLE_ID = "vehicle_id"
FRAME = "frame"
LANE_ID = "laneId"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"

# STATIC FILE
MEAN_X_VELOCITY = "meanXVelocity"
NUMBER_LANE_CHANGES = "numLaneChanges"
DRIVING_DIRECTION = "drivingDirection"
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
VEHICLE_CLASS = "vehicleClass"

# LC START TIME and END TIME
LC_START_GLOBAL_FRAME = "LC_start_global_frame"
LC_END_GLOBAL_FRAME = "LC_end_global_frame"
LC_START_LOCAL_FRAME = "LC_start_local_frame"
LC_END_LOCAL_FRAME = "LC_end_local_frame"
LC_MOMENT_LOCAL = "LC_moment_local"
LC_MOMENT_GLOBAL = "LC_moment_global"

# BL START MIDDLE END TIME
BL_START_GLOBAL_FRAME = "BL_start_global_frame"
BL_MIDDLE_GLOBAL_FRAME = "BL_middle_global_frame"
BL_END_GLOBAL_FRAME = "BL_end_global_frame"
BL_START_LOCAL_FRAME = "BL_start_local_frame"
BL_MIDDLE_LOCAL_FRAME = "BL_middle_local_frame"
BL_END_LOCAL_FRAME = "BL_end_local_frame"
IS_COMPLETE_START = "is_complete_start"
IS_COMPLETE_END = "is_complete_end"

# VIDEO MATE
SPEED_LIMIT = "speedLimit"
UPPER_LANE_MARKINGS = "upperLaneMarkings"
LOWER_LANE_MARKINGS = "lowerLaneMarkings"
FRAME_RATE = "frameRate"


LC_INTERACTION="LC_interaction"
LC_ACTION_START_LOCAL_FRAME="LC_action_start_local_frame"
LC_ACTION_END_LOCAL_FRAME="LC_action_end_local_frame"
LC_ACTION_START_GLOBAL_FRAME="LC_action_start_global_frame"
LC_ACTION_END_GLOBAL_FRAME="LC_action_end_global_frame"

# INTERACTION ATTRIBUTE IN LANE CHANGE OF INTENTION
INTENTION_EGO_PRE_S_DISTANCE="intention_ego_preceding_S_distance"
INTENTION_EGO_LTCP_S_DISTANCE="intention_ego_LTCpreceding_S_distance"
INTENTION_EGO_LTCF_S_DISTANCE="intention_ego_LTCfollowing_S_distance"
INTENTION_GAP="intention_gap"
INTENTION_EGO_IN_GAP_NORMALIZATION="intention_ego_in_gap_normalization"
INTENTION_EGO_PRE_S_VELOCITY="intention_ego_preceding_S_velocity"
INTENTION_EGO_LTCP_S_VELOCITY="intention_ego_LTCpreceding_S_velocity"
INTENTION_EGO_LTCF_S_VELOCITY="intention_ego_LTCfollowing_S_velocity"
INTENTION_TTC="intention_ttc"
INTENTION_LTC_FORWARD_TTC="intention_LTC_forward_TTC"
INTENTION_LTC_BACKWARD_TTC="intention_LTC_backward_TTC"

# INTERACTION ATTRIBUTE IN LANE CHANGE OF ACTION
ACTION_EGO_PRE_S_DISTANCE="action_ego_preceding_S_distance"
ACTION_EGO_LTCP_S_DISTANCE="action_ego_LTCpreceding_S_distance"
ACTION_EGO_LTCF_S_DISTANCE="action_ego_LTCfollowing_S_distance"
ACTION_GAP="action_gap"
ACTION_EGO_IN_GAP_NORMALIZATION="action_ego_in_gap_normalization"
ACTION_EGO_PRE_S_VELOCITY="action_ego_preceding_S_velocity"
ACTION_EGO_LTCP_S_VELOCITY="action_ego_LTCpreceding_S_velocity"
ACTION_EGO_LTCF_S_VELOCITY="action_ego_LTCfollowing_S_velocity"
ACTION_TTC="action_ttc"
ACTION_LTC_FORWARD_TTC="action_LTC_forward_TTC"
ACTION_LTC_BACKWARD_TTC="action_LTC_backward_TTC"

# INTERACTION ATTRIBUTE IN LANE CHANGE MOMENT
LCMOMENT_EGO_PRE_S_DISTANCE="LCmoment_ego_preceding_S_distance"
LCMOMENT_EGO_LTCP_S_DISTANCE="LCmoment_ego_LTCpreceding_S_distance"
LCMOMENT_EGO_LTCF_S_DISTANCE="LCmoment_ego_LTCfollowing_S_distance"
LCMOMENT_GAP="LCmoment_gap"
LCMOMENT_EGO_IN_GAP_NORMALIZATION="LCmoment_ego_in_gap_normalization"
LCMOMENT_EGO_PRE_S_VELOCITY="LCmoment_ego_preceding_S_velocity"
LCMOMENT_EGO_LTCP_S_VELOCITY="LCmoment_ego_LTCpreceding_S_velocity"
LCMOMENT_EGO_LTCF_S_VELOCITY="LCmoment_ego_LTCfollowing_S_velocity"
LCMOMENT_TTC="LCmoment_ttc"
LCMOMENT_LTC_FORWARD_TTC="LCmoment_LTC_forward_TTC"
LCMOMENT_LTC_BACKWARD_TTC="LCmoment_LTC_backward_TTC"
 
# GLOBAL VARIABLE
UPPER_LANE_WIDTH_DICT=None
LOWER_LANE_WIDTH_DICT=None
VEHICLE_IN_EACH_FRAME=None
EPSILON=0.0000001
# SPEED_LIMIT_EACH_FEATURE_ID_2=[]

# SUB-FUNCTION

def generate_vehile_in_each_frame(tracks):
    global VEHICLE_IN_EACH_FRAME
    VEHICLE_IN_EACH_FRAME={}
    for vehicle_instance in tracks:
        frame=vehicle_instance[FRAME]
        for key in frame:
            if not VEHICLE_IN_EACH_FRAME.get(key):
                VEHICLE_IN_EACH_FRAME[key]=[vehicle_instance["id"]]
            else:
                VEHICLE_IN_EACH_FRAME[key].append(vehicle_instance["id"])
    return 1

def lane_change_moment(lane)->int:
    assert isinstance(lane, np.ndarray)
    if len(lane)<2:
        return 0
    lo=0
    hi=len(lane)-1
    while lo+1<hi:
        mi=int(lo+(hi-lo)/2)
        if lane[mi]==lane[lo]:
            lo=mi
        else:
            hi=mi
    
    return hi

def scene_average_speed(static_info,static,key)->np.float32:
    Direction=int(static[DRIVING_DIRECTION])
    speed_mean=static[MEAN_X_VELOCITY]
    speed_count=1
    middle_frame=static[INITIAL_FRAME]+(static[FINAL_FRAME]-static[INITIAL_FRAME])/2
    #left key
    left_key=key-1
    while static_info.get(left_key):
        if int(static_info[left_key][DRIVING_DIRECTION])!=Direction:
            left_key-=1
            continue
        
        current_middle_frame=static_info[left_key][INITIAL_FRAME] + (static_info[left_key][FINAL_FRAME]-static_info[left_key][INITIAL_FRAME])/2
        if not (middle_frame<=current_middle_frame+500 and current_middle_frame-500<=middle_frame):
            left_key-=1
            break
        
        speed_count+=1
        speed_mean+=static_info[left_key][MEAN_X_VELOCITY]
        left_key-=1
    #right key
    right_key=key+1
    while static_info.get(right_key):
        if int(static_info[right_key][DRIVING_DIRECTION])!=Direction:
            right_key+=1
            continue
        
        current_middle_frame=static_info[right_key][INITIAL_FRAME] + (static_info[right_key][FINAL_FRAME]-static_info[right_key][INITIAL_FRAME])/2
        if not (middle_frame<=current_middle_frame+500 and current_middle_frame-500<=middle_frame):
            right_key+=1
            break
        
        speed_count+=1
        speed_mean+=static_info[right_key][MEAN_X_VELOCITY]
        right_key+=1
        
    speed_mean=speed_mean/speed_count        
    
    return speed_mean

def get_current_lane_average_speed(tracks,instance,static_info,static,key)->np.ndarray:
    Direction=int(static[DRIVING_DIRECTION])
    negetive_flag= -1 if Direction == 1 else 1
    currentlane=instance[LANE_ID]
    current_lane_average_speed=np.zeros((len(currentlane),),dtype=np.float32)
    
    for i in range(len(currentlane)):
        speed_mean=instance[X_VELOCITY][i]*negetive_flag
        speed_count=1
        current_frame=instance[FRAME][i]
        current_frame_vehicle=VEHICLE_IN_EACH_FRAME[current_frame]
        
        for vehicle in current_frame_vehicle:
            if int(static_info[vehicle][DRIVING_DIRECTION]) == Direction and tracks[vehicle-1][LANE_ID][current_frame-tracks[vehicle-1][FRAME][0]]==currentlane[i]:
                speed_mean+=tracks[vehicle-1][X_VELOCITY][current_frame-tracks[vehicle-1][FRAME][0]]*negetive_flag
                speed_count+=1
         
        current_lane_average_speed[i]=speed_mean/speed_count
        
    return current_lane_average_speed

def get_current_lane_width(instance,meta,Direction)->np.ndarray:
    
    global UPPER_LANE_WIDTH_DICT
    global LOWER_LANE_WIDTH_DICT
    
    if not UPPER_LANE_WIDTH_DICT and Direction == 1:
        lane_markings=meta[UPPER_LANE_MARKINGS]
        lane_width=np.zeros((len(lane_markings)-1,),dtype=np.float32)
        for i in range(len(lane_markings)-1):
            lane_width[i]=lane_markings[i+1]-lane_markings[i]
        
        init_lane=instance[LANE_ID][0]
        init_d_location=instance[BBOX][0,1]+instance[BBOX][0,3]/2
        first_lane_id=None
        
        for i in range(len(lane_markings)-1):
            if lane_markings[i]<init_d_location and init_d_location<=lane_markings[i+1]:
                first_lane_id=init_lane-i
                break
            
        assert first_lane_id
        
        UPPER_LANE_WIDTH_DICT={}
        for i in range(len(lane_markings)-1):
            UPPER_LANE_WIDTH_DICT[first_lane_id+i]=lane_width[i]
    
    elif not LOWER_LANE_WIDTH_DICT and Direction == 2:
        lane_markings=meta[LOWER_LANE_MARKINGS]
        lane_width=np.zeros((len(lane_markings)-1,),dtype=np.float32)
        for i in range(len(lane_markings)-1):
            lane_width[i]=lane_markings[i+1]-lane_markings[i]
        
        init_lane=instance[LANE_ID][0]
        init_d_location=instance[BBOX][0,1]+instance[BBOX][0,3]/2
        first_lane_id=None
        
        for i in range(len(lane_markings)-1):
            if lane_markings[i]<init_d_location and init_d_location<=lane_markings[i+1]:
                first_lane_id=init_lane-i
                break
            
        assert first_lane_id
        
        LOWER_LANE_WIDTH_DICT={}
        for i in range(len(lane_markings)-1):
            LOWER_LANE_WIDTH_DICT[first_lane_id+i]=lane_width[i]
    # end if        
    total_frame_len=len(instance[FRAME])
    current_lane_width=np.zeros((total_frame_len,),dtype=np.float32)
    temp_current_lane_width_dict=UPPER_LANE_WIDTH_DICT if Direction == 1 else LOWER_LANE_WIDTH_DICT
    
    for idx in range(total_frame_len):
        current_lane_width[idx]=temp_current_lane_width_dict[instance[LANE_ID][idx]]
    
    return current_lane_width

def get_total_number_lane(Direction)->np.uint8:
    number_of_total_lane=None
    if Direction == 1 and UPPER_LANE_WIDTH_DICT:
        number_of_total_lane=len(UPPER_LANE_WIDTH_DICT)
    elif Direction == 2 and LOWER_LANE_WIDTH_DICT:
        number_of_total_lane=len(LOWER_LANE_WIDTH_DICT)
    assert isinstance(number_of_total_lane, int)
    
    return number_of_total_lane

def left_right_maneuver_judgement(instance,Direction,maneuver)->np.uint8:
    init_lane=instance[LANE_ID][0]
    if maneuver == 0:
        final_lane=instance[LANE_ID][len(instance[FRAME])-1]
    else:
        for lane in instance[LANE_ID]:
            if init_lane != lane:
                final_lane = lane
                break
            
    assert not init_lane==final_lane
    left_right_maneuver=None#left:0  right:1
    if Direction == 1:
        left_right_maneuver=0 if init_lane<final_lane else 1
    else:
        left_right_maneuver=1 if init_lane<final_lane else 0
    
    return np.uint8(left_right_maneuver)

def get_feature_lane_id(instance,Direction)->np.ndarray:
    fetaure_lane_id=None
    if Direction==1:
        all_key=[key for key,_ in UPPER_LANE_WIDTH_DICT.items()]
        min_lane_id=np.min(all_key)
        fetaure_lane_id=instance[LANE_ID]-min_lane_id+1
    else:
        all_key=[key for key,_ in LOWER_LANE_WIDTH_DICT.items()]
        max_lane_id=np.max(all_key)
        fetaure_lane_id=max_lane_id-instance[LANE_ID]+1
        
    return fetaure_lane_id

def get_limit_speed(total_num_lane,global_speed_limit)->list:
    limit_speed_each_lane=[]
    if global_speed_limit==-1:
        if total_num_lane==2:
            limit_speed_each_lane=[30.0,40.0]
        elif total_num_lane==3:
            limit_speed_each_lane=[30.0,35.0,40.0]
        else:
            assert 0
    else:
        if total_num_lane==2:
            limit_speed_each_lane=[global_speed_limit*3/4,global_speed_limit]
        elif total_num_lane==3:
            limit_speed_each_lane=[global_speed_limit*3/4,global_speed_limit*7/8,global_speed_limit]
        else:
            assert 0
            
    return limit_speed_each_lane

def get_current_lane_speed_limit(lane_feature_id,limit_speed_each_lane)->np.ndarray:
    current_lane_speed_limit=np.zeros((len(lane_feature_id),),dtype=np.float32)
    for idx,lane_id in enumerate(lane_feature_id):
        current_lane_speed_limit[idx]=limit_speed_each_lane[lane_id-1]
        
    return current_lane_speed_limit

def get_distance_to_lane(s_location,meta,intention_lane,Direction,left_right_maneuver)->np.ndarray:
    if isinstance(left_right_maneuver, np.uint8):
        left_right_maneuver=np.zeros((len(intention_lane),),dtype=np.uint8)+left_right_maneuver
    
    target_lane_coor=meta[UPPER_LANE_MARKINGS] if Direction==1 else meta[LOWER_LANE_MARKINGS]
    ego_vehicle_loc=np.zeros((len(intention_lane),),dtype=np.float32)
    for idx,int_lane in enumerate(intention_lane):
        if Direction==1:
            ego_vehicle_loc[idx]=target_lane_coor[int_lane-1] if left_right_maneuver[idx] == 0 else target_lane_coor[int_lane]
            ego_vehicle_loc[idx]=s_location[idx,1]+s_location[idx,3]/2-ego_vehicle_loc[idx]
        else:
            ego_vehicle_loc[idx]=target_lane_coor[-1*int_lane] if left_right_maneuver[idx] == 0 else target_lane_coor[-1*int_lane-1]
            ego_vehicle_loc[idx]=-1*(s_location[idx,1]+s_location[idx,3]/2-ego_vehicle_loc[idx])
    
    return ego_vehicle_loc #left lane change: - -> +; right lane change + -> - 

def get_vehicle_s_location(Direction,instance)->np.ndarray: #offset
    return instance[BBOX][:,0] if Direction == 2 else 400-(instance[BBOX][:,0]+instance[BBOX][:,2])

def get_vehicle_d_location(Direction,instance)->np.ndarray: #offset
    return instance[BBOX][:,1]+instance[BBOX][:,3]/2 if Direction == 2 else 30-(instance[BBOX][:,1]+instance[BBOX][:,3]/2)

def get_vehicle_s_velocity(Direction,instance)->np.ndarray:
    return instance[X_VELOCITY] if Direction == 2 else -1*instance[X_VELOCITY]

def get_vehicle_d_velotity(Direction,instance)->np.ndarray:
    return instance[Y_VELOCITY] if Direction == 2 else -1*instance[Y_VELOCITY]

def get_vehicle_s_acceleration(Direction,instance)->np.ndarray:
    return instance[X_ACCELERATION] if Direction == 2 else -1*instance[X_ACCELERATION]

def get_vehicle_d_acceleration(Direction,instance)->np.ndarray:
    return instance[Y_ACCELERATION] if Direction == 2 else -1*instance[Y_ACCELERATION]

def get_vehicle_jerk(acceleration,meta)->np.ndarray:
    jerk=np.zeros((len(acceleration),),dtype=np.float32)
    for i in range(len(acceleration)-1):
        jerk[i]=(acceleration[i+1]-acceleration[i])/meta[FRAME_RATE]
    jerk[len(acceleration)-1]=jerk[len(acceleration)-2]
    return jerk

def get_lane_type(current_feature_lane_id,LEFT_RIGHT,total_num_lane)->np.ndarray:# 1 single solid ,2 double solid, 3 single dotted
    lane_type=np.zeros((len(current_feature_lane_id),),dtype=np.uint8)
    if LEFT_RIGHT:#left 0, right 1
        for idx,lane_id in enumerate(current_feature_lane_id):
            lane_type[idx] = 1 if lane_id == 1 else 3
    else:
        for idx,lane_id in enumerate(current_feature_lane_id):
            lane_type[idx] = 2 if lane_id == total_num_lane else 3
    
    return lane_type

def get_steering_rate_entropy(heading_angle)->np.float32:
    sre=0
    if len(heading_angle)<10:
        return sre
    error=np.zeros(len(heading_angle),dtype=np.float32)
    for i in range(3,len(heading_angle)):
        error[i]=2.5*heading_angle[i-1]-2*heading_angle[i-2]+0.5*heading_angle[i-3]-heading_angle[i]
    error[0]=error[3]
    error[1]=error[3]
    error[2]=error[3]
    
    count=np.zeros((9,),dtype=np.float32)
    for i in range(len(heading_angle)):
        if 0<=(error[i]+0.00045)/0.0001 and (error[i]+0.00045)/0.0001<9:
            count[int((error[i]+0.00045)/0.0001)]+=1
    
    count=count/np.sum(count)
    
    for i in range(9):
        sre+=-count[i]*math.log(count[i]+EPSILON,9)

    return sre

def get_vehicle_feature(vehicle_id,tracks,instance,static_info,LC_moment,PREFOL_LTC,features_of_single_LC)->dict:
    frame=instance[FRAME]
    total_frame_len=len(frame)
    if isinstance(LC_moment, np.ndarray) and len(LC_moment) == 2:
        assert frame[0]<=LC_moment[0] and LC_moment[1]<=frame[total_frame_len-1]
        LC_moment=LC_moment[1]
    else:
        assert frame[0]<=LC_moment and LC_moment<=frame[total_frame_len-1]
        
    surrounding_vehicle_feature={}
    surrounding_vehicle_feature[CLASS]=np.zeros((total_frame_len,),dtype=np.uint8)
    surrounding_vehicle_feature[VEHICLE_WIDTH]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[VEHICLE_HEIGHT]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[S_LOCATION]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[D_LOCATION]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[S_VELOCITY]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[D_VELOCITY]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[S_ACCELERATION]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[D_ACCELERATION]=np.zeros((total_frame_len,),dtype=np.float32)
    # tmp=features_of_single_LC[IS_PRECEDING_VEHICLE]
    for i in range(total_frame_len):
        if vehicle_id[i] == 0:
            continue
        if PREFOL_LTC and i+frame[0]>LC_moment:
            break
        global_frame=i+frame[0]
        surrounding_instance = tracks[vehicle_id[i]-1]
        surrounding_static = static_info[vehicle_id[i]]
        Direction=int(surrounding_static[DRIVING_DIRECTION])
        surrounding_vehicle_feature[CLASS][i]=1 if surrounding_static[CLASS] == "Car" else 2
        surrounding_vehicle_feature[VEHICLE_WIDTH][i]=surrounding_instance[BBOX][0,3].astype(np.float32)
        surrounding_vehicle_feature[VEHICLE_HEIGHT][i]=surrounding_instance[BBOX][0,2].astype(np.float32)

        surrounding_vehicle_feature[S_LOCATION][i]=(surrounding_instance[BBOX][global_frame-surrounding_instance[FRAME][0],0]).astype(np.float32) \
            if Direction == 2 else 400.0-(surrounding_instance[BBOX][global_frame-surrounding_instance[FRAME][0],0]+surrounding_instance[BBOX][0,2]).astype(np.float32)
        surrounding_vehicle_feature[D_LOCATION][i]=(surrounding_instance[BBOX][global_frame-surrounding_instance[FRAME][0],1]+surrounding_instance[BBOX][0,3]/2).astype(np.float32) \
            if Direction == 2 else 30.0-(surrounding_instance[BBOX][global_frame-surrounding_instance[FRAME][0],1]+surrounding_instance[BBOX][0,3]/2).astype(np.float32)
        surrounding_vehicle_feature[S_VELOCITY][i]=surrounding_instance[X_VELOCITY][global_frame-surrounding_instance[FRAME][0]].astype(np.float32) \
            if Direction == 2 else -1*surrounding_instance[X_VELOCITY][global_frame-surrounding_instance[FRAME][0]].astype(np.float32)
        surrounding_vehicle_feature[D_VELOCITY][i]=surrounding_instance[Y_VELOCITY][global_frame-surrounding_instance[FRAME][0]].astype(np.float32) \
            if Direction == 2 else -1*surrounding_instance[Y_VELOCITY][global_frame-surrounding_instance[FRAME][0]].astype(np.float32)
        surrounding_vehicle_feature[S_ACCELERATION][i]=surrounding_instance[X_ACCELERATION][global_frame-surrounding_instance[FRAME][0]].astype(np.float32) \
            if Direction == 2 else -1*surrounding_instance[X_ACCELERATION][global_frame-surrounding_instance[FRAME][0]].astype(np.float32)
        surrounding_vehicle_feature[D_ACCELERATION][i]=surrounding_instance[Y_ACCELERATION][global_frame-surrounding_instance[FRAME][0]].astype(np.float32) \
            if Direction == 2 else -1*surrounding_instance[Y_ACCELERATION][global_frame-surrounding_instance[FRAME][0]].astype(np.float32)        
    
    return surrounding_vehicle_feature

def get_gap(LC_moment,instance,tracks,features_of_single_LC,Direction)->np.ndarray:
    global_frame=instance[FRAME]
    local_LC_moment=LC_moment-global_frame[0]
    LC_preceding=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment+1]
    LC_following=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment+1]   
    gap=np.zeros((len(global_frame),),dtype=np.float32)
    
    for i in range(len(global_frame)):
        if i+global_frame[0]>LC_moment:
            break
        LC_preceding_instance=tracks[LC_preceding-1]
        LC_following_instance=tracks[LC_following-1]
        current_global_frame=i+global_frame[0]
        LC_preceding_S_loc=-1
        LC_following_S_loc=-1
        if LC_preceding_instance[FRAME][0]<=current_global_frame and current_global_frame<=LC_preceding_instance[FRAME][len(LC_preceding_instance[FRAME])-1]:
            LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]
        if LC_following_instance[FRAME][0]<=current_global_frame and current_global_frame<=LC_following_instance[FRAME][len(LC_following_instance[FRAME])-1]:
            LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]
        
        if (LC_preceding_S_loc==-1) or (LC_following_S_loc==-1):
            gap[i]=200
        else:
            gap[i]=np.abs(LC_preceding_S_loc+LC_preceding_instance[BBOX][0,2]-LC_following_S_loc) if Direction == 1 \
                else np.abs(LC_preceding_S_loc-LC_preceding_instance[BBOX][0,2]-LC_following_S_loc)
        
    return gap

def get_gap_BL(LC_moment,instance,tracks,features_of_single_LC,Direction)->np.ndarray:
    global_frame=instance[FRAME]
    local_LC_moment=LC_moment-global_frame[0]
    
    LC_preceding_0=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment[0]+1]
    LC_following_0=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment[0]+1]
    
    LC_preceding_1=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment[1]+1]
    LC_following_1=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment[1]+1]  
    
    gap=np.zeros((len(global_frame),),dtype=np.float32)
    
    for i in range(len(global_frame)):
        if i+global_frame[0]>LC_moment[0]:
            break
        LC_preceding_instance=tracks[LC_preceding_0-1]
        LC_following_instance=tracks[LC_following_0-1]
        current_global_frame=i+global_frame[0]
        LC_preceding_S_loc=-1
        LC_following_S_loc=-1
        if LC_preceding_instance[FRAME][0]<=current_global_frame and current_global_frame<=LC_preceding_instance[FRAME][len(LC_preceding_instance[FRAME])-1]:
            LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]
        if LC_following_instance[FRAME][0]<=current_global_frame and current_global_frame<=LC_following_instance[FRAME][len(LC_following_instance[FRAME])-1]:
            LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]
        
        if (LC_preceding_S_loc==-1) or (LC_following_S_loc==-1):
            gap[i]=200
        else:
            gap[i]=np.abs(LC_preceding_S_loc+LC_preceding_instance[BBOX][0,2]-LC_following_S_loc) if Direction == 1 \
                else np.abs(LC_preceding_S_loc-LC_preceding_instance[BBOX][0,2]-LC_following_S_loc)
    
    for i in range(len(global_frame)):
        if i+global_frame[0]<int((LC_moment[0]+LC_moment[1])/2):
            continue
        if i+global_frame[0]>LC_moment[1]:
            break
        LC_preceding_instance=tracks[LC_preceding_1-1]
        LC_following_instance=tracks[LC_following_1-1]
        current_global_frame=i+global_frame[0]
        LC_preceding_S_loc=-1
        LC_following_S_loc=-1
        if LC_preceding_instance[FRAME][0]<=current_global_frame and current_global_frame<=LC_preceding_instance[FRAME][len(LC_preceding_instance[FRAME])-1]:
            LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]
        if LC_following_instance[FRAME][0]<=current_global_frame and current_global_frame<=LC_following_instance[FRAME][len(LC_following_instance[FRAME])-1]:
            LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]
        
        if (LC_preceding_S_loc==-1) or (LC_following_S_loc==-1):
            gap[i]=200
        else:
            gap[i]=np.abs(LC_preceding_S_loc+LC_preceding_instance[BBOX][0,2]-LC_following_S_loc) if Direction == 1 \
                else np.abs(LC_preceding_S_loc-LC_preceding_instance[BBOX][0,2]-LC_following_S_loc)
    
    return gap

def get_forward_LTC_thwdhwttc(LC_moment_global, instance, tracks, features_of_single_LC, Direction, THWDHWTTC)->np.ndarray:
    global_frame=instance[FRAME]
    local_LC_moment=LC_moment_global-global_frame[0]
    LC_preceding=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment+1]
    # LC_following=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment+1]   
    rtn=np.zeros((len(global_frame),),dtype=np.float32)+200.0
    
    for i in range(len(global_frame)):
        # if i+global_frame[0]>LC_moment_global:
        #     break
        current_global_frame=i+global_frame[0]
        if THWDHWTTC == 1:#THW
            if LC_preceding != 0 and LC_preceding in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height))/np.abs(instance[X_VELOCITY][i]+EPSILON)
        
        elif THWDHWTTC == 2:#DHW
            if LC_preceding != 0 and LC_preceding in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height))
                    
        else:#TTC
            if LC_preceding != 0 and LC_preceding in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                LC_preceding_S_velocity=LC_preceding_instance[X_VELOCITY][current_global_frame-LC_preceding_instance[FRAME][0]]
                if (LC_preceding_S_loc-(current_S_loc+current_height))>=0:              
                    rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height)) / (-instance[X_VELOCITY][i]+LC_preceding_S_velocity+EPSILON)  if Direction == 1\
                        else (LC_preceding_S_loc-(current_S_loc+current_height)) / (instance[X_VELOCITY][i]-LC_preceding_S_velocity+EPSILON)
    return rtn

def get_forward_LTC_thwdhwttc_BL(LC_moment_global, instance, tracks, features_of_single_LC, Direction, THWDHWTTC)->np.ndarray:
    global_frame=instance[FRAME]
    local_LC_moment=LC_moment_global-global_frame[0]
    LC_preceding_0=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment[0]+1]
    LC_preceding_1=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment[1]+1]
    # LC_following=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment+1]   
    rtn=np.zeros((len(global_frame),),dtype=np.float32)+200.0
    
    for i in range(len(global_frame)):
        if i+global_frame[0]>int((LC_moment_global[0]+LC_moment_global[1])/2):
            break
        current_global_frame=i+global_frame[0]
        if THWDHWTTC == 1:#THW
            if LC_preceding_0 != 0 and LC_preceding_0 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding_0-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height))/np.abs(instance[X_VELOCITY][i]+EPSILON)
        
        elif THWDHWTTC == 2:#DHW
            if LC_preceding_0 != 0 and LC_preceding_0 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding_0-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height))
                    
        else:#TTC
            if LC_preceding_0 != 0 and LC_preceding_0 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding_0-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                LC_preceding_S_velocity=LC_preceding_instance[X_VELOCITY][current_global_frame-LC_preceding_instance[FRAME][0]]
                if (LC_preceding_S_loc-(current_S_loc+current_height))>=0:              
                    rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height)) / (-instance[X_VELOCITY][i]+LC_preceding_S_velocity+EPSILON)  if Direction == 1\
                        else (LC_preceding_S_loc-(current_S_loc+current_height)) / (instance[X_VELOCITY][i]-LC_preceding_S_velocity+EPSILON)

    for i in range(len(global_frame)):
        if i+global_frame[0]<=int((LC_moment_global[0]+LC_moment_global[1])/2):
            continue
        current_global_frame=i+global_frame[0]
        if THWDHWTTC == 1:#THW
            if LC_preceding_1 != 0 and LC_preceding_1 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding_1-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height))/np.abs(instance[X_VELOCITY][i]+EPSILON)
        
        elif THWDHWTTC == 2:#DHW
            if LC_preceding_1 != 0 and LC_preceding_1 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding_1-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height))
                    
        else:#TTC
            if LC_preceding_1 != 0 and LC_preceding_1 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_preceding_instance=tracks[LC_preceding_1-1]
                LC_preceding_S_loc=LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_preceding_instance[BBOX][current_global_frame-LC_preceding_instance[FRAME][0],0]+LC_preceding_instance[BBOX][0,2])
                current_height=instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                LC_preceding_S_velocity=LC_preceding_instance[X_VELOCITY][current_global_frame-LC_preceding_instance[FRAME][0]]
                if (LC_preceding_S_loc-(current_S_loc+current_height))>=0:              
                    rtn[i]=(LC_preceding_S_loc-(current_S_loc+current_height)) / (-instance[X_VELOCITY][i]+LC_preceding_S_velocity+EPSILON)  if Direction == 1\
                        else (LC_preceding_S_loc-(current_S_loc+current_height)) / (instance[X_VELOCITY][i]-LC_preceding_S_velocity+EPSILON)

    return rtn

def get_backward_LTC_thwhdwttc(LC_moment_global, instance, tracks, features_of_single_LC, Direction, THWDHWTTC)->np.ndarray:
    global_frame=instance[FRAME]
    local_LC_moment=LC_moment_global-global_frame[0]
    # LC_preceding=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment+1]
    LC_following=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment+1]   
    rtn=np.zeros((len(global_frame),),dtype=np.float32)+200.0
    
    for i in range(len(global_frame)):
        # if i+global_frame[0]>LC_moment_global:
        #     break
        current_global_frame=i+global_frame[0]
        if THWDHWTTC == 1:#THW
            if LC_following != 0 and LC_following in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(current_S_loc-(LC_following_S_loc+following_height))/np.abs(instance[X_VELOCITY][i]+EPSILON)
        
        elif THWDHWTTC == 2:#DHW
            if LC_following != 0 and LC_following in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(current_S_loc-(LC_following_S_loc+following_height))
                    
        else:#TTC
            if LC_following != 0 and LC_following in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                LC_following_S_velocity=LC_following_instance[X_VELOCITY][current_global_frame-LC_following_instance[FRAME][0]]
                if (current_S_loc-(LC_following_S_loc+following_height))>=0:              
                    rtn[i]=(current_S_loc-(LC_following_S_loc+following_height)) / (-LC_following_S_velocity+instance[X_VELOCITY][i]+EPSILON)  if Direction == 1\
                        else (current_S_loc-(LC_following_S_loc+following_height)) / (LC_following_S_velocity-instance[X_VELOCITY][i]+EPSILON)
    return rtn

def get_backward_LTC_thwhdwttc_BL(LC_moment_global, instance, tracks, features_of_single_LC, Direction, THWDHWTTC)->np.ndarray:
    global_frame=instance[FRAME]
    local_LC_moment=LC_moment_global-global_frame[0]
    # LC_preceding=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment+1]
    LC_following_0=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment[0]+1] 
    LC_following_1=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment[1]+1] 
    rtn=np.zeros((len(global_frame),),dtype=np.float32)+200.0
    
    for i in range(len(global_frame)):
        if i+global_frame[0]>int((LC_moment_global[0]+LC_moment_global[1])/2):
            break
        current_global_frame=i+global_frame[0]
        if THWDHWTTC == 1:#THW
            if LC_following_0 != 0 and LC_following_0 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following_0-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(current_S_loc-(LC_following_S_loc+following_height))/np.abs(instance[X_VELOCITY][i]+EPSILON)
        
        elif THWDHWTTC == 2:#DHW
            if LC_following_0 != 0 and LC_following_0 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following_0-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(current_S_loc-(LC_following_S_loc+following_height))
                    
        else:#TTC
            if LC_following_0 != 0 and LC_following_0 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following_0-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                LC_following_S_velocity=LC_following_instance[X_VELOCITY][current_global_frame-LC_following_instance[FRAME][0]]
                if (current_S_loc-(LC_following_S_loc+following_height))>=0:              
                    rtn[i]=(current_S_loc-(LC_following_S_loc+following_height)) / (-LC_following_S_velocity+instance[X_VELOCITY][i]+EPSILON)  if Direction == 1\
                        else (current_S_loc-(LC_following_S_loc+following_height)) / (LC_following_S_velocity-instance[X_VELOCITY][i]+EPSILON)
 
    for i in range(len(global_frame)):
        if i+global_frame[0]<=int((LC_moment_global[0]+LC_moment_global[1])/2):
            continue
        current_global_frame=i+global_frame[0]
        if THWDHWTTC == 1:#THW
            if LC_following_1 != 0 and LC_following_1 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following_1-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(current_S_loc-(LC_following_S_loc+following_height))/np.abs(instance[X_VELOCITY][i]+EPSILON)
        
        elif THWDHWTTC == 2:#DHW
            if LC_following_1 != 0 and LC_following_1 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following_1-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                rtn[i]=(current_S_loc-(LC_following_S_loc+following_height))
                    
        else:#TTC
            if LC_following_1 != 0 and LC_following_1 in VEHICLE_IN_EACH_FRAME[global_frame[i]]:
                LC_following_instance=tracks[LC_following_1-1]
                LC_following_S_loc=LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0] if Direction==2\
                    else 400-(LC_following_instance[BBOX][current_global_frame-LC_following_instance[FRAME][0],0]+LC_following_instance[BBOX][0,2])
                following_height=LC_following_instance[BBOX][0,2]
                current_S_loc=features_of_single_LC[S_LOCATION][i]
                
                LC_following_S_velocity=LC_following_instance[X_VELOCITY][current_global_frame-LC_following_instance[FRAME][0]]
                if (current_S_loc-(LC_following_S_loc+following_height))>=0:              
                    rtn[i]=(current_S_loc-(LC_following_S_loc+following_height)) / (-LC_following_S_velocity+instance[X_VELOCITY][i]+EPSILON)  if Direction == 1\
                        else (current_S_loc-(LC_following_S_loc+following_height)) / (LC_following_S_velocity-instance[X_VELOCITY][i]+EPSILON)

    return rtn

def borrow_lane_moment(lane)->list:
    assert isinstance(lane, np.ndarray)
    if len(lane)<2:
        return [0,1]
    current_lane=lane[0]
    BL_moment=[]
    for i in range(len(lane)):
        if current_lane!=lane[i]:
            current_lane=lane[i]
            BL_moment.append(i)
    
    assert len(BL_moment)==2

    return BL_moment

def get_intention_lane(current_feature_lane_id,BL_middle_localidx,left_right_maneuver)->np.ndarray:#Borrow lane intention
    intention_lane=np.zeros((len(current_feature_lane_id),),dtype=np.uint8)
    intention_lane[0:BL_middle_localidx]+=current_feature_lane_id[BL_middle_localidx]
    intention_lane[BL_middle_localidx:]+=current_feature_lane_id[len(current_feature_lane_id)-1]
    
    bl_left_right_maneuver=np.zeros((len(current_feature_lane_id),),dtype=np.uint8)
    bl_left_right_maneuver[0:BL_middle_localidx]+=left_right_maneuver
    bl_left_right_maneuver[BL_middle_localidx:]+=(not left_right_maneuver)
    
    return intention_lane,bl_left_right_maneuver

def get_LC_start_moment_local(features_of_single_LC,LC_moment_local):
    # is_complete_start=True
    # LC_start_localidx=None
    flag=1
    if features_of_single_LC[LEFT_RIGHT_MANEUVER]==0:
        flag=-1
    vehicle_acceleration=features_of_single_LC[D_ACCELERATION]*flag
    
    LC_start_moment_local_right_boundary=max(0,LC_moment_local-1*25)
    if LC_moment_local-1*25<=0:
        return 0,False    
    LC_start_moment_local_left_boundary=max(0,LC_moment_local-8*25)
    middle_frame=int((LC_start_moment_local_right_boundary+LC_start_moment_local_left_boundary)/2)
    acc_01_moment=None
    if 0.1<=vehicle_acceleration[middle_frame]:
        acc_01_moment=middle_frame
        while(acc_01_moment>=LC_start_moment_local_left_boundary):
            if vehicle_acceleration[acc_01_moment]<0.1:
                break
            else:
                acc_01_moment-=1
        acc_01_moment+=1
        
        if acc_01_moment <= LC_start_moment_local_left_boundary:
            if 0<=LC_moment_local-8*25:
                return LC_moment_local-8*25,True
            else:
                return 0,False
    else:
        acc_01_moment=min(middle_frame+1,len(vehicle_acceleration))
        while acc_01_moment<=LC_start_moment_local_right_boundary:
            if vehicle_acceleration[acc_01_moment]>=0.1:
                break
            else:
                acc_01_moment+=1
        if acc_01_moment > LC_start_moment_local_right_boundary:
            if np.abs(features_of_single_LC[D_VELOCITY][acc_01_moment])>=0.2:
                return LC_start_moment_local_left_boundary,False
            else: 
                return LC_start_moment_local_left_boundary,False
    
    while LC_start_moment_local_left_boundary<=acc_01_moment:
        if vehicle_acceleration[acc_01_moment]<0.03:
            break
        else:
            acc_01_moment-=1
    AA_LC_start_localidx=max(0,acc_01_moment-5)
    if np.abs(features_of_single_LC[D_VELOCITY][AA_LC_start_localidx])>=0.2:
        return AA_LC_start_localidx,False
    else:
        return AA_LC_start_localidx,True# -5 is a compensate for jerk

def get_LC_end_moment_local(features_of_single_LC,LC_moment_local):
    # is_complete_start=True
    # LC_start_localidx=None
    flag=1
    if features_of_single_LC[LEFT_RIGHT_MANEUVER]!=0:
        flag=-1
    vehicle_acceleration=features_of_single_LC[D_ACCELERATION]*flag
    vehicle_velocity=features_of_single_LC[D_VELOCITY]*flag*(-1)
    
    LC_end_moment_local_right_boundary=min(len(vehicle_acceleration),LC_moment_local+1*25)
    if LC_moment_local+1*25>=len(vehicle_acceleration):
        return len(vehicle_acceleration)-1,False    
    LC_end_moment_local_left_boundary=min(len(vehicle_acceleration),LC_moment_local+8*25)
    middle_frame=int((LC_end_moment_local_right_boundary+LC_end_moment_local_left_boundary)/2)
    acc_01_moment=None
    if 0.1<=vehicle_velocity[middle_frame]:
        acc_01_moment=middle_frame
        while acc_01_moment<LC_end_moment_local_left_boundary:
            if vehicle_velocity[acc_01_moment]<0.1:
                break
            else:
                acc_01_moment+=1
        acc_01_moment-=1
        
        if acc_01_moment >= LC_end_moment_local_left_boundary:
            if LC_moment_local+8*25 <= LC_end_moment_local_left_boundary:
                return LC_moment_local+8*25,True
            else:
                return len(vehicle_velocity)-1,False
    else:
        acc_01_moment=max(middle_frame-1,0)
        while acc_01_moment>=LC_end_moment_local_right_boundary:
            if vehicle_velocity[acc_01_moment]>=0.1:
                break
            else:
                acc_01_moment-=1
    
    while LC_end_moment_local_left_boundary>acc_01_moment:
        if vehicle_velocity[acc_01_moment]<0.03:
            break
        else:
            acc_01_moment+=1
    AA_LC_start_localidx=min(len(vehicle_acceleration)-1,acc_01_moment+10)
    if np.abs(features_of_single_LC[D_ACCELERATION][AA_LC_start_localidx])>=0.2 or np.abs(features_of_single_LC[D_VELOCITY][AA_LC_start_localidx])>0.1:
        return AA_LC_start_localidx,False
    else:
        return AA_LC_start_localidx,True# -5 is a compensate for jerk

def get_LC_action_start_local(features_of_single_LC,LC_start_localidx,is_complete_start):
    flag=1
    if features_of_single_LC[LEFT_RIGHT_MANEUVER]!=0:
        flag=-1
    vehicle_acceleration=features_of_single_LC[D_ACCELERATION]*flag*(-1)
    vehicle_velocity=features_of_single_LC[D_VELOCITY]*flag*(-1)
    total_len=len(vehicle_velocity)
    
    final_idx=None
    
    if is_complete_start:
        origin_location=features_of_single_LC[D_LOCATION][LC_start_localidx]
        final_idx_loc=LC_start_localidx
        for i in range(total_len-LC_start_localidx):
            diff=np.abs(features_of_single_LC[D_LOCATION][LC_start_localidx+i]-origin_location)
            if diff >= 0.1:
                final_idx_loc=LC_start_localidx+i
                break
            if i == total_len-LC_start_localidx-1:
                return LC_start_localidx,False
        
        final_idx_vel=LC_start_localidx
        for i in range(total_len-LC_start_localidx):
            if vehicle_velocity[LC_start_localidx+i]>=0.1:
                final_idx_vel=LC_start_localidx+i
                break
            if i == total_len-LC_start_localidx-1:
                return LC_start_localidx,False
        
        final_idx=int((final_idx_loc+final_idx_vel)/2)  
    else:
        final_idx_vel=LC_start_localidx
        if vehicle_velocity[LC_start_localidx]>=0.1:
            return LC_start_localidx,False
            
        for i in range(total_len-LC_start_localidx):
            if vehicle_velocity[LC_start_localidx+i]>=0.1:
                final_idx_vel=LC_start_localidx+i
                break
            if i == total_len-LC_start_localidx:
                return LC_start_localidx,False
            
        final_idx=max(0,final_idx_vel-10)
    
    return final_idx,True

def get_LC_action_end_local(features_of_single_LC,LC_end_localidx,is_complete_end):
    flag=1
    if features_of_single_LC[LEFT_RIGHT_MANEUVER]!=0:
        flag=-1
    vehicle_acceleration=features_of_single_LC[D_ACCELERATION]*flag
    vehicle_velocity=features_of_single_LC[D_VELOCITY]*flag*(-1)
    total_len=len(vehicle_velocity)
    
    final_idx=None
    
    if is_complete_end:
        origin_location=features_of_single_LC[D_LOCATION][LC_end_localidx]
        final_idx_loc=LC_end_localidx
        for i in range(LC_end_localidx):
            diff=np.abs(features_of_single_LC[D_LOCATION][LC_end_localidx-i]-origin_location)
            if diff >= 0.1:
                final_idx_loc=LC_end_localidx-i
                break
            if i == LC_end_localidx-1:
                return LC_end_localidx,False
        
        final_idx_vel=LC_end_localidx
        for i in range(LC_end_localidx):
            if vehicle_velocity[LC_end_localidx-i]>=0.1:
                final_idx_vel=LC_end_localidx-i
                break
            if i == LC_end_localidx-1:
                return LC_end_localidx,False
        
        final_idx=int((final_idx_loc+final_idx_vel)/2)  
    else:
        final_idx_vel=LC_end_localidx
        if vehicle_velocity[LC_end_localidx]>=0.1:
            return LC_end_localidx,False
            
        for i in range(LC_end_localidx):
            if vehicle_velocity[LC_end_localidx-i]>=0.1:
                final_idx_vel=LC_end_localidx-i
                break
            if i == LC_end_localidx-1:
                return LC_end_localidx,False
            
        final_idx=min(total_len-1,final_idx_vel+10)
    
    return final_idx,True

def get_intention_action_moment_attribute(LC_moment_global,instance,features_of_single_LC,tracks,Direction,
                                          LC_start_localidx,is_complete_start,LC_start_global_frame,
                                          LC_action_start_localidx,is_complete_action_start,LC_action_start_global_frame):
    global_frame=instance[FRAME]
    local_LC_moment=LC_moment_global-global_frame[0]
    LC_preceding=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment+1]
    LC_preceding_instance=tracks[LC_preceding-1] if LC_preceding!=0 else None 
    LC_following=features_of_single_LC[IS_FOLLOWING_VEHICLE][local_LC_moment+1] 
    LC_following_instance=tracks[LC_following-1] if LC_following!=0 else None
    preceding=features_of_single_LC[IS_PRECEDING_VEHICLE][local_LC_moment]
    preceding_instance=tracks[preceding-1] if preceding!=0 else None
    if LC_preceding_instance:
        LC_pre_localidx=LC_start_global_frame-LC_preceding_instance[FRAME][0]
    is_LC_pre_localidx= (LC_preceding_instance and 0<=LC_pre_localidx and LC_pre_localidx<len(LC_preceding_instance[FRAME]))
    if LC_following_instance:    
        LC_fol_localidx=LC_start_global_frame-LC_following_instance[FRAME][0]
    is_LC_fol_localidx= (LC_following_instance and 0<=LC_fol_localidx and LC_fol_localidx<len(LC_following_instance[FRAME]))
    if preceding_instance:  
        pre_localidx=LC_start_global_frame-preceding_instance[FRAME][0]
    is_pre_localidx= (preceding_instance and 0<=pre_localidx and pre_localidx<len(preceding_instance[FRAME]))    
    LC_interaction={}
        
    # intention attribute
    if is_complete_start:   
        if Direction==1:
            intention_LC_pre_tail_S=LC_preceding_instance[BBOX][LC_pre_localidx,0]+LC_preceding_instance[BBOX][LC_pre_localidx,2] if is_LC_pre_localidx else -1000
            intention_LC_pre_SV=LC_preceding_instance[X_VELOCITY][LC_pre_localidx] if is_LC_pre_localidx else 0
            
            intention_pre_tail_S=preceding_instance[BBOX][pre_localidx,0]+preceding_instance[BBOX][pre_localidx,2] if  is_pre_localidx else -1000
            intention_pre_SV=preceding_instance[X_VELOCITY][pre_localidx] if is_pre_localidx else 0
                        
            intention_LC_fol_head_S=LC_following_instance[BBOX][LC_fol_localidx,0] if is_LC_fol_localidx else 1400
            intention_LC_fol_SV=LC_following_instance[X_VELOCITY][LC_fol_localidx] if is_LC_fol_localidx else 0
            
            current_vehicle_head_S=instance[BBOX][LC_start_localidx,0]
            current_vehicle_tail_S=instance[BBOX][LC_start_localidx,0]+instance[BBOX][LC_start_localidx,2]
            current_vehicle_SV=instance[X_VELOCITY][LC_start_localidx]
            
            LC_interaction[INTENTION_EGO_PRE_S_DISTANCE]=-1*(intention_pre_tail_S-current_vehicle_head_S) if is_pre_localidx else None
            LC_interaction[INTENTION_EGO_LTCP_S_DISTANCE]=-1*(intention_LC_pre_tail_S-current_vehicle_head_S) if is_LC_pre_localidx else None
            LC_interaction[INTENTION_EGO_LTCF_S_DISTANCE]=-1*(current_vehicle_tail_S-intention_LC_fol_head_S) if is_LC_fol_localidx else None
            LC_interaction[INTENTION_GAP]=-1*(intention_LC_pre_tail_S-intention_LC_fol_head_S)  if is_LC_pre_localidx and is_LC_fol_localidx else None
            LC_interaction[INTENTION_EGO_IN_GAP_NORMALIZATION]=LC_interaction[INTENTION_EGO_LTCF_S_DISTANCE]/(LC_interaction[INTENTION_GAP]+EPSILON) \
                                                                if is_LC_pre_localidx and is_LC_fol_localidx else None
            LC_interaction[INTENTION_EGO_PRE_S_VELOCITY]=(intention_pre_SV-current_vehicle_SV) if is_pre_localidx else None
            LC_interaction[INTENTION_EGO_LTCP_S_VELOCITY]=(intention_LC_pre_SV-current_vehicle_SV) if is_LC_pre_localidx else None
            LC_interaction[INTENTION_EGO_LTCF_S_VELOCITY]=-1*(current_vehicle_SV-intention_LC_fol_SV) if is_LC_fol_localidx else None
            
            LC_interaction[INTENTION_TTC]=LC_interaction[INTENTION_EGO_PRE_S_DISTANCE]/(LC_interaction[INTENTION_EGO_PRE_S_VELOCITY]+EPSILON) if is_pre_localidx else None
            LC_interaction[INTENTION_LTC_FORWARD_TTC]=LC_interaction[INTENTION_EGO_LTCP_S_DISTANCE]/(LC_interaction[INTENTION_EGO_LTCP_S_VELOCITY]+EPSILON) if is_LC_pre_localidx else None
            LC_interaction[INTENTION_LTC_BACKWARD_TTC]=-1*LC_interaction[INTENTION_EGO_LTCF_S_DISTANCE]/(LC_interaction[INTENTION_EGO_LTCF_S_VELOCITY]+EPSILON) if is_LC_fol_localidx else None
            
        else:
            intention_LC_pre_tail_S=LC_preceding_instance[BBOX][LC_pre_localidx,0] if is_LC_pre_localidx else -1000
            intention_LC_pre_SV=LC_preceding_instance[X_VELOCITY][LC_pre_localidx] if is_LC_pre_localidx else 0
            
            intention_pre_tail_S=preceding_instance[BBOX][pre_localidx,0] if is_pre_localidx else -1000
            intention_pre_SV=preceding_instance[X_VELOCITY][pre_localidx] if is_pre_localidx else 0
                        
            intention_LC_fol_head_S=LC_following_instance[BBOX][LC_fol_localidx,0]+LC_following_instance[BBOX][LC_fol_localidx,2] if is_LC_fol_localidx else 1400
            intention_LC_fol_SV=LC_following_instance[X_VELOCITY][LC_fol_localidx] if is_LC_fol_localidx else 0
            
            current_vehicle_head_S=instance[BBOX][LC_start_localidx,0]+instance[BBOX][LC_start_localidx,2]
            current_vehicle_tail_S=instance[BBOX][LC_start_localidx,0]
            current_vehicle_SV=instance[X_VELOCITY][LC_start_localidx,]
            
            LC_interaction[INTENTION_EGO_PRE_S_DISTANCE]=(intention_pre_tail_S-current_vehicle_head_S) if is_pre_localidx else None
            LC_interaction[INTENTION_EGO_LTCP_S_DISTANCE]=(intention_LC_pre_tail_S-current_vehicle_head_S) if is_LC_pre_localidx else None
            LC_interaction[INTENTION_EGO_LTCF_S_DISTANCE]=(current_vehicle_tail_S-intention_LC_fol_head_S) if is_LC_fol_localidx else None
            LC_interaction[INTENTION_GAP]=(intention_LC_pre_tail_S-intention_LC_fol_head_S)  if is_LC_fol_localidx and is_LC_pre_localidx else None
            LC_interaction[INTENTION_EGO_IN_GAP_NORMALIZATION]=LC_interaction[INTENTION_EGO_LTCF_S_DISTANCE]/(LC_interaction[INTENTION_GAP]+EPSILON) \
                                                                if is_LC_fol_localidx and is_LC_pre_localidx else None
            LC_interaction[INTENTION_EGO_PRE_S_VELOCITY]=-1*(intention_pre_SV-current_vehicle_SV) if is_pre_localidx else None
            LC_interaction[INTENTION_EGO_LTCP_S_VELOCITY]=-1*(intention_LC_pre_SV-current_vehicle_SV) if is_LC_pre_localidx else None
            LC_interaction[INTENTION_EGO_LTCF_S_VELOCITY]=(current_vehicle_SV-intention_LC_fol_SV) if is_LC_fol_localidx else None  

            LC_interaction[INTENTION_TTC]=LC_interaction[INTENTION_EGO_PRE_S_DISTANCE]/(LC_interaction[INTENTION_EGO_PRE_S_VELOCITY]+EPSILON) if is_pre_localidx else None
            LC_interaction[INTENTION_LTC_FORWARD_TTC]=LC_interaction[INTENTION_EGO_LTCP_S_DISTANCE]/(LC_interaction[INTENTION_EGO_LTCP_S_VELOCITY]+EPSILON) if is_LC_pre_localidx else None
            LC_interaction[INTENTION_LTC_BACKWARD_TTC]=-1*LC_interaction[INTENTION_EGO_LTCF_S_DISTANCE]/(LC_interaction[INTENTION_EGO_LTCF_S_VELOCITY]+EPSILON) if is_LC_fol_localidx else None
 
        
    else:
        LC_interaction[INTENTION_EGO_PRE_S_DISTANCE]=None
        LC_interaction[INTENTION_EGO_LTCP_S_DISTANCE]=None
        LC_interaction[INTENTION_EGO_LTCF_S_DISTANCE]=None
        LC_interaction[INTENTION_GAP]=None
        LC_interaction[INTENTION_EGO_IN_GAP_NORMALIZATION]=None
        LC_interaction[INTENTION_EGO_PRE_S_VELOCITY]=None
        LC_interaction[INTENTION_EGO_LTCP_S_VELOCITY]=None
        LC_interaction[INTENTION_EGO_LTCF_S_VELOCITY]=None
        LC_interaction[INTENTION_TTC]=None
        LC_interaction[INTENTION_LTC_FORWARD_TTC]=None
        LC_interaction[INTENTION_LTC_BACKWARD_TTC]=None
     
    if LC_preceding_instance:
        LC_pre_localidx=LC_action_start_global_frame-LC_preceding_instance[FRAME][0]
    is_LC_pre_localidx= (LC_preceding_instance and 0<=LC_pre_localidx and LC_pre_localidx<len(LC_preceding_instance[FRAME]))
    if LC_following_instance:    
        LC_fol_localidx=LC_action_start_global_frame-LC_following_instance[FRAME][0]
    is_LC_fol_localidx= (LC_following_instance and 0<=LC_fol_localidx and LC_fol_localidx<len(LC_following_instance[FRAME]))
    if preceding_instance:  
        pre_localidx=LC_action_start_global_frame-preceding_instance[FRAME][0]
    is_pre_localidx= (preceding_instance and 0<=pre_localidx and pre_localidx<len(preceding_instance[FRAME]))  
    
    # action attribute
    if is_complete_start:
        if Direction==1:
            action_LC_pre_tail_S=LC_preceding_instance[BBOX][LC_pre_localidx,0]+LC_preceding_instance[BBOX][LC_pre_localidx,2] if is_LC_pre_localidx else -1000
            action_LC_pre_SV=LC_preceding_instance[X_VELOCITY][LC_pre_localidx] if is_LC_pre_localidx else 0
            
            action_pre_tail_S=preceding_instance[BBOX][pre_localidx,0]+preceding_instance[BBOX][pre_localidx,2] if  is_pre_localidx else -1000
            action_pre_SV=preceding_instance[X_VELOCITY][pre_localidx] if is_pre_localidx else 0
                        
            action_LC_fol_head_S=LC_following_instance[BBOX][LC_fol_localidx,0] if is_LC_fol_localidx else 1400
            action_LC_fol_SV=LC_following_instance[X_VELOCITY][LC_fol_localidx] if is_LC_fol_localidx else 0
            
            actioncurrent_vehicle_head_S=instance[BBOX][LC_action_start_localidx,0]
            actioncurrent_vehicle_tail_S=instance[BBOX][LC_action_start_localidx,0]+instance[BBOX][LC_action_start_localidx,2]
            actioncurrent_vehicle_SV=instance[X_VELOCITY][LC_action_start_localidx]
            
            LC_interaction[ACTION_EGO_PRE_S_DISTANCE]=-1*(action_pre_tail_S-actioncurrent_vehicle_head_S) if is_pre_localidx else None
            LC_interaction[ACTION_EGO_LTCP_S_DISTANCE]=-1*(action_LC_pre_tail_S-actioncurrent_vehicle_head_S) if is_LC_pre_localidx else None
            LC_interaction[ACTION_EGO_LTCF_S_DISTANCE]=-1*(actioncurrent_vehicle_tail_S-action_LC_fol_head_S) if is_LC_fol_localidx else None
            LC_interaction[ACTION_GAP]=-1*(action_LC_pre_tail_S-action_LC_fol_head_S)  if is_LC_pre_localidx and is_LC_fol_localidx else None
            LC_interaction[ACTION_EGO_IN_GAP_NORMALIZATION]=LC_interaction[ACTION_EGO_LTCF_S_DISTANCE]/(LC_interaction[ACTION_GAP]+EPSILON) \
                                                                if is_LC_pre_localidx and is_LC_fol_localidx else None
            LC_interaction[ACTION_EGO_PRE_S_VELOCITY]=(action_pre_SV-actioncurrent_vehicle_SV) if is_pre_localidx else None
            LC_interaction[ACTION_EGO_LTCP_S_VELOCITY]=(action_LC_pre_SV-actioncurrent_vehicle_SV) if is_LC_pre_localidx else None
            LC_interaction[ACTION_EGO_LTCF_S_VELOCITY]=-1*(actioncurrent_vehicle_SV-action_LC_fol_SV) if is_LC_fol_localidx else None

            LC_interaction[ACTION_TTC]=LC_interaction[ACTION_EGO_PRE_S_DISTANCE]/(LC_interaction[ACTION_EGO_PRE_S_VELOCITY]+EPSILON) if is_pre_localidx else None
            LC_interaction[ACTION_LTC_FORWARD_TTC]=LC_interaction[ACTION_EGO_LTCP_S_DISTANCE]/(LC_interaction[ACTION_EGO_LTCP_S_VELOCITY]+EPSILON) if is_LC_pre_localidx else None
            LC_interaction[ACTION_LTC_BACKWARD_TTC]=-1*LC_interaction[ACTION_EGO_LTCF_S_DISTANCE]/(LC_interaction[ACTION_EGO_LTCF_S_VELOCITY]+EPSILON) if is_LC_fol_localidx else None

        
        else:
            action_LC_pre_tail_S=LC_preceding_instance[BBOX][LC_pre_localidx,0] if is_LC_pre_localidx else -1000
            action_LC_pre_SV=LC_preceding_instance[X_VELOCITY][LC_pre_localidx] if is_LC_pre_localidx else 0
            
            action_pre_tail_S=preceding_instance[BBOX][pre_localidx,0] if is_pre_localidx else -1000
            action_pre_SV=preceding_instance[X_VELOCITY][pre_localidx] if is_pre_localidx else 0
                        
            action_LC_fol_head_S=LC_following_instance[BBOX][LC_fol_localidx,0]+LC_following_instance[BBOX][LC_fol_localidx,2] if is_LC_fol_localidx else 1400
            action_LC_fol_SV=LC_following_instance[X_VELOCITY][LC_fol_localidx] if is_LC_fol_localidx else 0
            
            actioncurrent_vehicle_head_S=instance[BBOX][LC_action_start_localidx,0]+instance[BBOX][LC_action_start_localidx,2]
            actioncurrent_vehicle_tail_S=instance[BBOX][LC_action_start_localidx,0]
            actioncurrent_vehicle_SV=instance[X_VELOCITY][LC_action_start_localidx]
            
            LC_interaction[ACTION_EGO_PRE_S_DISTANCE]=(action_pre_tail_S-actioncurrent_vehicle_head_S) if is_pre_localidx else None
            LC_interaction[ACTION_EGO_LTCP_S_DISTANCE]=(action_LC_pre_tail_S-actioncurrent_vehicle_head_S) if is_LC_pre_localidx else None
            LC_interaction[ACTION_EGO_LTCF_S_DISTANCE]=(actioncurrent_vehicle_tail_S-action_LC_fol_head_S) if is_LC_fol_localidx else None
            LC_interaction[ACTION_GAP]=(action_LC_pre_tail_S-action_LC_fol_head_S)  if is_LC_fol_localidx and is_LC_pre_localidx else None
            LC_interaction[ACTION_EGO_IN_GAP_NORMALIZATION]=LC_interaction[ACTION_EGO_LTCF_S_DISTANCE]/(LC_interaction[ACTION_GAP]+EPSILON) \
                                                                if is_LC_fol_localidx and is_LC_pre_localidx else None
            LC_interaction[ACTION_EGO_PRE_S_VELOCITY]=-1*(action_pre_SV-actioncurrent_vehicle_SV) if is_pre_localidx else None
            LC_interaction[ACTION_EGO_LTCP_S_VELOCITY]=-1*(action_LC_pre_SV-actioncurrent_vehicle_SV) if is_LC_pre_localidx else None
            LC_interaction[ACTION_EGO_LTCF_S_VELOCITY]=(actioncurrent_vehicle_SV-action_LC_fol_SV) if is_LC_fol_localidx else None  

            LC_interaction[ACTION_TTC]=LC_interaction[ACTION_EGO_PRE_S_DISTANCE]/(LC_interaction[ACTION_EGO_PRE_S_VELOCITY]+EPSILON) if is_pre_localidx else None
            LC_interaction[ACTION_LTC_FORWARD_TTC]=LC_interaction[ACTION_EGO_LTCP_S_DISTANCE]/(LC_interaction[ACTION_EGO_LTCP_S_VELOCITY]+EPSILON) if is_LC_pre_localidx else None
            LC_interaction[ACTION_LTC_BACKWARD_TTC]=-1*LC_interaction[ACTION_EGO_LTCF_S_DISTANCE]/(LC_interaction[ACTION_EGO_LTCF_S_VELOCITY]+EPSILON) if is_LC_fol_localidx else None
        
    else:
        LC_interaction[ACTION_EGO_PRE_S_DISTANCE]=None
        LC_interaction[ACTION_EGO_LTCP_S_DISTANCE]=None
        LC_interaction[ACTION_EGO_LTCF_S_DISTANCE]=None
        LC_interaction[ACTION_GAP]=None
        LC_interaction[ACTION_EGO_IN_GAP_NORMALIZATION]=None
        LC_interaction[ACTION_EGO_PRE_S_VELOCITY]=None
        LC_interaction[ACTION_EGO_LTCP_S_VELOCITY]=None
        LC_interaction[ACTION_EGO_LTCF_S_VELOCITY]=None
        LC_interaction[ACTION_TTC]=None
        LC_interaction[ACTION_LTC_FORWARD_TTC]=None
        LC_interaction[ACTION_LTC_BACKWARD_TTC]=None
    
    if LC_preceding_instance:
        LC_pre_localidx=LC_moment_global-LC_preceding_instance[FRAME][0]
    is_LC_pre_localidx= (LC_preceding_instance and 0<=LC_pre_localidx and LC_pre_localidx<len(LC_preceding_instance[FRAME]))
    if LC_following_instance:    
        LC_fol_localidx=LC_moment_global-LC_following_instance[FRAME][0]
    is_LC_fol_localidx= (LC_following_instance and 0<=LC_fol_localidx and LC_fol_localidx<len(LC_following_instance[FRAME]))
    if preceding_instance:  
        pre_localidx=LC_moment_global-preceding_instance[FRAME][0]
    is_pre_localidx= (preceding_instance and 0<=pre_localidx and pre_localidx<len(preceding_instance[FRAME]))  
    
    # LC moment attribute
    if is_complete_start:
        if Direction==1:
            LCmoment_LC_pre_tail_S=LC_preceding_instance[BBOX][LC_pre_localidx,0]+LC_preceding_instance[BBOX][LC_pre_localidx,2] if is_LC_pre_localidx else -1000
            LCmoment_LC_pre_SV=LC_preceding_instance[X_VELOCITY][LC_pre_localidx] if is_LC_pre_localidx else 0
            
            LCmoment_pre_tail_S=preceding_instance[BBOX][pre_localidx,0]+preceding_instance[BBOX][pre_localidx,2] if  is_pre_localidx else -1000
            LCmoment_pre_SV=preceding_instance[X_VELOCITY][pre_localidx] if is_pre_localidx else 0
                        
            LCmoment_LC_fol_head_S=LC_following_instance[BBOX][LC_fol_localidx,0] if is_LC_fol_localidx else 1400
            LCmoment_LC_fol_SV=LC_following_instance[X_VELOCITY][LC_fol_localidx] if is_LC_fol_localidx else 0
            
            LCmomentcurrent_vehicle_head_S=instance[BBOX][local_LC_moment,0]
            LCmomentcurrent_vehicle_tail_S=instance[BBOX][local_LC_moment,0]+instance[BBOX][local_LC_moment,2]
            LCmomentcurrent_vehicle_SV=instance[X_VELOCITY][local_LC_moment]
            
            LC_interaction[LCMOMENT_EGO_PRE_S_DISTANCE]=-1*(LCmoment_pre_tail_S-LCmomentcurrent_vehicle_head_S) if is_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_LTCP_S_DISTANCE]=-1*(LCmoment_LC_pre_tail_S-LCmomentcurrent_vehicle_head_S) if is_LC_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_LTCF_S_DISTANCE]=-1*(LCmomentcurrent_vehicle_tail_S-LCmoment_LC_fol_head_S) if is_LC_fol_localidx else None
            LC_interaction[LCMOMENT_GAP]=-1*(LCmoment_LC_pre_tail_S-LCmoment_LC_fol_head_S)  if is_LC_pre_localidx and is_LC_fol_localidx else None
            LC_interaction[LCMOMENT_EGO_IN_GAP_NORMALIZATION]=LC_interaction[LCMOMENT_EGO_LTCF_S_DISTANCE]/(LC_interaction[LCMOMENT_GAP]+EPSILON) \
                                                                if is_LC_pre_localidx and is_LC_fol_localidx else None
            LC_interaction[LCMOMENT_EGO_PRE_S_VELOCITY]=(LCmoment_pre_SV-LCmomentcurrent_vehicle_SV) if is_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_LTCP_S_VELOCITY]=(LCmoment_LC_pre_SV-LCmomentcurrent_vehicle_SV) if is_LC_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_LTCF_S_VELOCITY]=-1*(LCmomentcurrent_vehicle_SV-LCmoment_LC_fol_SV) if is_LC_fol_localidx else None

            LC_interaction[LCMOMENT_TTC]=LC_interaction[LCMOMENT_EGO_PRE_S_DISTANCE]/(LC_interaction[LCMOMENT_EGO_PRE_S_VELOCITY]+EPSILON) if is_pre_localidx else None
            LC_interaction[LCMOMENT_LTC_FORWARD_TTC]=LC_interaction[LCMOMENT_EGO_LTCP_S_DISTANCE]/(LC_interaction[LCMOMENT_EGO_LTCP_S_VELOCITY]+EPSILON) if is_LC_pre_localidx else None
            LC_interaction[LCMOMENT_LTC_BACKWARD_TTC]=-1*LC_interaction[LCMOMENT_EGO_LTCF_S_DISTANCE]/(LC_interaction[LCMOMENT_EGO_LTCF_S_VELOCITY]+EPSILON) if is_LC_fol_localidx else None
  
        
        else:
            LCmoment_LC_pre_tail_S=LC_preceding_instance[BBOX][LC_pre_localidx,0] if is_LC_pre_localidx else -1000
            LCmoment_LC_pre_SV=LC_preceding_instance[X_VELOCITY][LC_pre_localidx] if is_LC_pre_localidx else 0
            
            LCmoment_pre_tail_S=preceding_instance[BBOX][pre_localidx,0] if is_pre_localidx else -1000
            LCmoment_pre_SV=preceding_instance[X_VELOCITY][pre_localidx] if is_pre_localidx else 0
                        
            LCmoment_LC_fol_head_S=LC_following_instance[BBOX][LC_fol_localidx,0]+LC_following_instance[BBOX][LC_fol_localidx,2] if is_LC_fol_localidx else 1400
            LCmoment_LC_fol_SV=LC_following_instance[X_VELOCITY][LC_fol_localidx] if is_LC_fol_localidx else 0
            
            LCmomentcurrent_vehicle_head_S=instance[BBOX][local_LC_moment,0]+instance[BBOX][local_LC_moment,2]
            LCmomentcurrent_vehicle_tail_S=instance[BBOX][local_LC_moment,0]
            LCmomentcurrent_vehicle_SV=instance[X_VELOCITY][local_LC_moment]
            
            LC_interaction[LCMOMENT_EGO_PRE_S_DISTANCE]=(LCmoment_pre_tail_S-LCmomentcurrent_vehicle_head_S) if is_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_LTCP_S_DISTANCE]=(LCmoment_LC_pre_tail_S-LCmomentcurrent_vehicle_head_S) if is_LC_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_LTCF_S_DISTANCE]=(LCmomentcurrent_vehicle_tail_S-LCmoment_LC_fol_head_S) if is_LC_fol_localidx else None
            LC_interaction[LCMOMENT_GAP]=(LCmoment_LC_pre_tail_S-LCmoment_LC_fol_head_S)  if is_LC_fol_localidx and is_LC_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_IN_GAP_NORMALIZATION]=LC_interaction[LCMOMENT_EGO_LTCF_S_DISTANCE]/(LC_interaction[LCMOMENT_GAP]+EPSILON) \
                                                                if is_LC_fol_localidx and is_LC_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_PRE_S_VELOCITY]=-1*(LCmoment_pre_SV-LCmomentcurrent_vehicle_SV) if is_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_LTCP_S_VELOCITY]=-1*(LCmoment_LC_pre_SV-LCmomentcurrent_vehicle_SV) if is_LC_pre_localidx else None
            LC_interaction[LCMOMENT_EGO_LTCF_S_VELOCITY]=(LCmomentcurrent_vehicle_SV-LCmoment_LC_fol_SV) if is_LC_fol_localidx else None  

            LC_interaction[LCMOMENT_TTC]=LC_interaction[LCMOMENT_EGO_PRE_S_DISTANCE]/(LC_interaction[LCMOMENT_EGO_PRE_S_VELOCITY]+EPSILON) if is_pre_localidx else None
            LC_interaction[LCMOMENT_LTC_FORWARD_TTC]=LC_interaction[LCMOMENT_EGO_LTCP_S_DISTANCE]/(LC_interaction[LCMOMENT_EGO_LTCP_S_VELOCITY]+EPSILON) if is_LC_pre_localidx else None
            LC_interaction[LCMOMENT_LTC_BACKWARD_TTC]=-1*LC_interaction[LCMOMENT_EGO_LTCF_S_DISTANCE]/(LC_interaction[LCMOMENT_EGO_LTCF_S_VELOCITY]+EPSILON) if is_LC_fol_localidx else None

    else:
        LC_interaction[LCMOMENT_EGO_PRE_S_DISTANCE]=None
        LC_interaction[LCMOMENT_EGO_LTCP_S_DISTANCE]=None
        LC_interaction[LCMOMENT_EGO_LTCF_S_DISTANCE]=None
        LC_interaction[LCMOMENT_GAP]=None
        LC_interaction[LCMOMENT_EGO_IN_GAP_NORMALIZATION]=None
        LC_interaction[LCMOMENT_EGO_PRE_S_VELOCITY]=None
        LC_interaction[LCMOMENT_EGO_LTCP_S_VELOCITY]=None
        LC_interaction[LCMOMENT_EGO_LTCF_S_VELOCITY]=None
        LC_interaction[LCMOMENT_TTC]=None
        LC_interaction[LCMOMENT_LTC_FORWARD_TTC]=None
        LC_interaction[LCMOMENT_LTC_BACKWARD_TTC]=None    
    
    
    return LC_interaction

def get_feature(vehicle_id,tracks,instance,static_info):
    frame=instance[FRAME]
    total_frame_len=len(frame)
        
    surrounding_vehicle_feature={}
    surrounding_vehicle_feature[VEHICLE_ID]=vehicle_id
    surrounding_vehicle_feature[CLASS]=np.zeros((total_frame_len,),dtype=np.uint8)
    surrounding_vehicle_feature[S_LOCATION]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[D_LOCATION]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[S_VELOCITY]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[D_VELOCITY]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[S_ACCELERATION]=np.zeros((total_frame_len,),dtype=np.float32)
    surrounding_vehicle_feature[D_ACCELERATION]=np.zeros((total_frame_len,),dtype=np.float32)
    # tmp=features_of_single_LC[IS_PRECEDING_VEHICLE]
    for i in range(total_frame_len):
        if vehicle_id[i] == 0:
            continue
        global_frame=i+frame[0]
        surrounding_instance = tracks[vehicle_id[i]-1]
        surrounding_static = static_info[vehicle_id[i]]
        Direction=int(surrounding_static[DRIVING_DIRECTION])
        surrounding_vehicle_feature[CLASS][i]=1 if surrounding_static[CLASS] == "Car" else -1
        
        surrounding_vehicle_feature[S_LOCATION][i]=(surrounding_instance[BBOX][global_frame-surrounding_instance[FRAME][0],0]).astype(np.float32) \
            if Direction == 2 else 400.0-(surrounding_instance[BBOX][global_frame-surrounding_instance[FRAME][0],0]+surrounding_instance[BBOX][0,2]).astype(np.float32)
        surrounding_vehicle_feature[D_LOCATION][i]=(surrounding_instance[BBOX][global_frame-surrounding_instance[FRAME][0],1]+surrounding_instance[BBOX][0,3]/2).astype(np.float32) \
            if Direction == 2 else 30.0-(surrounding_instance[BBOX][global_frame-surrounding_instance[FRAME][0],1]+surrounding_instance[BBOX][0,3]/2).astype(np.float32)
        surrounding_vehicle_feature[S_VELOCITY][i]=surrounding_instance[X_VELOCITY][global_frame-surrounding_instance[FRAME][0]].astype(np.float32) \
            if Direction == 2 else -1*surrounding_instance[X_VELOCITY][global_frame-surrounding_instance[FRAME][0]].astype(np.float32)
        surrounding_vehicle_feature[D_VELOCITY][i]=surrounding_instance[Y_VELOCITY][global_frame-surrounding_instance[FRAME][0]].astype(np.float32) \
            if Direction == 2 else -1*surrounding_instance[Y_VELOCITY][global_frame-surrounding_instance[FRAME][0]].astype(np.float32)
        surrounding_vehicle_feature[S_ACCELERATION][i]=surrounding_instance[X_ACCELERATION][global_frame-surrounding_instance[FRAME][0]].astype(np.float32) \
            if Direction == 2 else -1*surrounding_instance[X_ACCELERATION][global_frame-surrounding_instance[FRAME][0]].astype(np.float32)
        surrounding_vehicle_feature[D_ACCELERATION][i]=surrounding_instance[Y_ACCELERATION][global_frame-surrounding_instance[FRAME][0]].astype(np.float32) \
            if Direction == 2 else -1*surrounding_instance[Y_ACCELERATION][global_frame-surrounding_instance[FRAME][0]].astype(np.float32)        
    
    return surrounding_vehicle_feature    


# MAIN_FUNCTION
def get_value_in_dict(*vehicle_feature_dict):
    rtn=None
    for vdict in vehicle_feature_dict:
        for _,items in vdict.items():
            items=np.reshape(items, (-1,1))
            if rtn is None:
                rtn=items
            else:
                rtn=np.concatenate((rtn,items),axis=1)
    
    return rtn
def extraction_trajectory_prediction_feature(tracks,static_info,meta_dict)->dict:
    assert isinstance(tracks, list) and len(tracks) != 0
    assert isinstance(static_info, dict) and len(tracks) == len(static_info)
    assert isinstance(meta_dict, dict) 
    trackslen=len(tracks)
    finalframe=tracks[trackslen-1][FRAME][-1]-500
    initframe=500
    return_final=None
    for idx, instance in enumerate(tracks):
        if tracks[idx][FRAME][0]<initframe or tracks[idx][FRAME][-1]>finalframe:
            continue
        
        frame=instance[FRAME]
        total_frame_len=len(frame)
        
        ego=get_feature(np.zeros((total_frame_len,),dtype=np.uint8)+idx+1,tracks,instance,static_info)
        pre=get_feature(instance[PRECEDING_ID],tracks,instance,static_info)
        fol=get_feature(instance[FOLLOWING_ID], tracks, instance, static_info)
        lftpre=get_feature(instance[LEFT_PRECEDING_ID], tracks, instance, static_info)
        lftalo=get_feature(instance[LEFT_ALONGSIDE_ID], tracks, instance, static_info)
        lftfol=get_feature(instance[LEFT_FOLLOWING_ID], tracks, instance, static_info)
        rgtpre=get_feature(instance[RIGHT_PRECEDING_ID], tracks, instance, static_info)
        rgtalo=get_feature(instance[RIGHT_ALONGSIDE_ID], tracks, instance, static_info)
        rgtfol=get_feature(instance[RIGHT_FOLLOWING_ID], tracks, instance, static_info)        
        lat=np.zeros((total_frame_len,),dtype=np.uint8)
        if static_info[idx+1][NUMBER_LANE_CHANGES] == 1:
            try:
                return_trajectory=extraction_in(tracks,static_info,meta_dict,idx+1)
            except:
                print("error1")
                continue
            
            if len(return_trajectory)<=0:
                continue
            LC_action_start_local_frame=return_trajectory[idx+1][LC_ACTION_START_LOCAL_FRAME]
            LC_action_end_local_frame=return_trajectory[idx+1][LC_ACTION_END_LOCAL_FRAME]
            #left =1 right=2
            lat[LC_action_start_local_frame:LC_action_end_local_frame]=1 if return_trajectory[idx+1][LEFT_RIGHT_MANEUVER]==0 else 2         
        elif static_info[idx+1][NUMBER_LANE_CHANGES] == 2:
            return_trajectory=extraction_BL_in(tracks,static_info,meta_dict,idx+1)
            if len(return_trajectory)<=0:
                continue
            BL_action_start_local_frame=return_trajectory[idx+1][BL_START_LOCAL_FRAME]
            BL_action_middle_local_frame=return_trajectory[idx+1][BL_MIDDLE_LOCAL_FRAME]
            BL_action_end_local_frame=return_trajectory[idx+1][BL_END_LOCAL_FRAME]
            lat[BL_action_start_local_frame:BL_action_middle_local_frame]=1 if return_trajectory[idx+1][LEFT_RIGHT_MANEUVER]==0 else 2      
            lat[BL_action_middle_local_frame:BL_action_end_local_frame]=2 if return_trajectory[idx+1][LEFT_RIGHT_MANEUVER]==0 else 1
        elif static_info[idx+1][NUMBER_LANE_CHANGES] >= 3:
            continue

        lon=np.zeros((total_frame_len,),dtype=np.uint8)
        for i in range(total_frame_len):
            if ego[S_ACCELERATION][i]>=0.5:
                lo=max(i-25,0)
                hi=min(total_frame_len,i+25)
                lon[lo:hi]=1
            elif ego[S_ACCELERATION][i]<=-0.5:
                lo=max(i-25,0)
                hi=min(total_frame_len,i+25)
                lon[lo:hi]=2
        
        rtn=get_value_in_dict(ego,pre,fol,lftpre,lftalo,lftfol,rgtpre,rgtalo,rgtfol)       
        lon=np.reshape(lon,(-1,1))
        lat=np.reshape(lat,(-1,1))
        frame=np.reshape(frame,(-1,1))
        rtn=np.concatenate((rtn,lat,lon,frame),axis=1)
        lo=0
        hi=total_frame_len
        for i in range(total_frame_len):
            if lo==0 and rtn[i,2]>=50 and rtn[i,-1]%5==0:
                lo=i
            if hi==total_frame_len and rtn[i,2]>=300 and rtn[i,-1]%5==0:
                hi=i
        hi+=1
        mask=np.zeros((total_frame_len,),dtype=bool)
        for i in range(total_frame_len):
            if i>=lo and i<=hi and rtn[i,-1]%5==0:
                mask[i]=True
        rtn=rtn[mask,:].astype(np.float32)
        if return_final is None:
            return_final=rtn
        else:
            return_final=np.vstack((return_final,rtn))
    
    return np.around(return_final,6)

def extraction_in(tracks,static_info,meta_dict,curid)->dict:
    # Assert
    assert isinstance(tracks, list) and len(tracks) != 0
    assert isinstance(static_info, dict) and len(tracks) == len(static_info)
    assert isinstance(meta_dict, dict)
    
    # Initialization
    return_trajectory = {}
    trajectory_count=0
    generate_vehile_in_each_frame(tracks)    
    
    for idx, instance in enumerate(tracks):
        # Preliminary judgment of LC
        
        if curid!=idx+1:
            continue    
        
        if static_info[idx+1][NUMBER_LANE_CHANGES] != 1:
            continue
        # Basic info: instance, static, meta
        static=static_info[idx+1]
        meta=meta_dict
        
        # Accurate judgment of LC
        total_frame_len=len(instance[FRAME])
        init_lane=instance[LANE_ID][0]
        final_lane=instance[LANE_ID][total_frame_len-1]
        # ---Condition1: If the initial lane and the final lane remain unchanged, 
        # ---            it is not a lane change, although it may be a combination of two lane changes. 
        if init_lane == final_lane:
            continue
        # It can make more precision results, but more difficulty. Current version is a initial version.
        
        # extract the moment of across lane
        LC_moment_local=lane_change_moment(instance[LANE_ID])#global
        LC_moment_global=LC_moment_local+instance[FRAME][0]-1
        # LC_start_localidx=max(0,LC_moment_local-100)
        # LC_end_localidx=min(len(instance[LANE_ID])-1,LC_moment_local+100)
        # LC_start_global_frame=int(instance[FRAME][LC_start_localidx])
        # LC_end_global_frame=int(instance[FRAME][LC_end_localidx])
        
        Direction=int(static[DRIVING_DIRECTION])
        
        features_of_single_LC={}
        features_of_single_LC[VEHICLE_ID]=np.uint16(idx+1)
        
        #Basic information of dataset (global information)
        features_of_single_LC[DATASET]=0
        features_of_single_LC[SCENE]=np.uint8(0) #0:highway 1:...
        features_of_single_LC[SCENE_AVERAGE_SPEED]=np.float32(scene_average_speed(static_info,static,idx+1))#unit: m/s
        
        #Local environment information of target (ego) vehicle
        features_of_single_LC[CURRENT_LANE_WIDTH]=get_current_lane_width(instance,meta,Direction).astype(np.float32)
        features_of_single_LC[TOTAL_NUM_LANE]=np.uint8(get_total_number_lane(Direction))
        features_of_single_LC[ROAD_SPEED_LIMIT]=np.float32(meta[SPEED_LIMIT]/3.6) if meta[SPEED_LIMIT]!=-1 else np.float32(100)#unit: m/s if no speed limit, default 100
        features_of_single_LC[CURRENT_ROAD_CURVATURE]=np.zeros((total_frame_len,),dtype=np.float32)+0.0001 #default 0.0001
        features_of_single_LC[TRAFFIC_LIGHT]=0
        
        #Driving maneuver classification
        features_of_single_LC[CURRENT_DRIVING_MANEUVER]=np.uint8(0)#0:lane change 1:borrow lane        
        features_of_single_LC[LEFT_RIGHT_MANEUVER]=left_right_maneuver_judgement(instance,Direction,features_of_single_LC[CURRENT_DRIVING_MANEUVER])
        features_of_single_LC[CURRENT_LANE_FEATURE_ID]=get_feature_lane_id(instance,Direction)
        features_of_single_LC[CURRENT_LANE_AVERAGE_SPEED]=get_current_lane_average_speed(tracks,instance,static_info,static,idx+1)
        features_of_single_LC[CURRENT_LANE_SPEED_LIMIT]=get_current_lane_speed_limit(features_of_single_LC[CURRENT_LANE_FEATURE_ID],get_limit_speed(features_of_single_LC[TOTAL_NUM_LANE],meta[SPEED_LIMIT]))
        features_of_single_LC[INTENTION_LANE]=np.zeros((total_frame_len,),dtype=np.uint8)+features_of_single_LC[CURRENT_LANE_FEATURE_ID][total_frame_len-1]
        
        # Target vehicle information
        features_of_single_LC[DISTANCE_TO_LANE]=get_distance_to_lane(instance[BBOX],meta,features_of_single_LC[INTENTION_LANE],Direction,features_of_single_LC[LEFT_RIGHT_MANEUVER])#offset
        features_of_single_LC[CLASS]=1 if static[CLASS] == "Car" else 2
        features_of_single_LC[VEHICLE_WIDTH]=instance[BBOX][0,3].astype(np.float32)
        features_of_single_LC[VEHICLE_HEIGHT]=instance[BBOX][0,2].astype(np.float32)
        features_of_single_LC[S_LOCATION]=get_vehicle_s_location(Direction,instance).astype(np.float32)#offset
        features_of_single_LC[D_LOCATION]=get_vehicle_d_location(Direction, instance).astype(np.float32)#offset
        features_of_single_LC[S_VELOCITY]=get_vehicle_s_velocity(Direction, instance).astype(np.float32)
        features_of_single_LC[D_VELOCITY]=get_vehicle_d_velotity(Direction, instance).astype(np.float32)
        features_of_single_LC[HEADING_ANGLE]=np.arctan(features_of_single_LC[D_VELOCITY]/(features_of_single_LC[S_VELOCITY]+EPSILON)).astype(np.float32) # Heading angle along the current lane 
        features_of_single_LC[S_ACCELERATION]=get_vehicle_s_acceleration(Direction, instance).astype(np.float32)
        features_of_single_LC[D_ACCELERATION]=get_vehicle_d_acceleration(Direction, instance).astype(np.float32)
        features_of_single_LC[YAW_RATE]=np.arctan(features_of_single_LC[D_ACCELERATION]/(features_of_single_LC[S_ACCELERATION]+EPSILON)).astype(np.float32)
        features_of_single_LC[S_JERK]=get_vehicle_jerk(features_of_single_LC[S_ACCELERATION], meta).astype(np.float32)
        features_of_single_LC[D_JERK]=get_vehicle_jerk(features_of_single_LC[D_ACCELERATION], meta).astype(np.float32)
        features_of_single_LC[TTC]=instance[TTC].astype(np.float32)
        features_of_single_LC[THW]=instance[THW].astype(np.float32)
        features_of_single_LC[DHW]=instance[DHW].astype(np.float32)
        features_of_single_LC[LEFT_LANE_TYPE]=get_lane_type(features_of_single_LC[CURRENT_LANE_FEATURE_ID], 0, features_of_single_LC[TOTAL_NUM_LANE])
        features_of_single_LC[RIGHT_LANE_TYPE]=get_lane_type(features_of_single_LC[CURRENT_LANE_FEATURE_ID], 1, features_of_single_LC[TOTAL_NUM_LANE])
        features_of_single_LC[STEERING_RATE_ENTROPY]=get_steering_rate_entropy(features_of_single_LC[HEADING_ANGLE])
        
        #Interactive information of surrounding vehicles
        features_of_single_LC[IS_PRECEDING_VEHICLE]=instance[PRECEDING_ID].astype(np.uint16)
        features_of_single_LC[IS_FOLLOWING_VEHICLE]=instance[FOLLOWING_ID].astype(np.uint16)
        features_of_single_LC[IS_LANE_TO_CHANGE_PRECEDING_VEHICLE]=instance[RIGHT_PRECEDING_ID].astype(np.uint16) if features_of_single_LC[LEFT_RIGHT_MANEUVER] else instance[LEFT_PRECEDING_ID].astype(np.uint16)
        features_of_single_LC[IS_LANE_TO_CHANGE_ALONGSIDE_VEHICLE]=instance[RIGHT_ALONGSIDE_ID].astype(np.uint16) if features_of_single_LC[LEFT_RIGHT_MANEUVER] else instance[LEFT_ALONGSIDE_ID].astype(np.uint16)
        features_of_single_LC[IS_LANE_TO_CHANGE_FOLLOWING_VEHICLE]=instance[RIGHT_FOLLOWING_ID].astype(np.uint16) if features_of_single_LC[LEFT_RIGHT_MANEUVER] else instance[LEFT_FOLLOWING_ID].astype(np.uint16)      
        features_of_single_LC[PRECEDING_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_LC[IS_PRECEDING_VEHICLE],tracks,instance,static_info,LC_moment_global,0,features_of_single_LC) #offset
        features_of_single_LC[FOLLOWING_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_LC[IS_FOLLOWING_VEHICLE],tracks,instance,static_info,LC_moment_global,0,features_of_single_LC) #offset
        features_of_single_LC[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_LC[IS_LANE_TO_CHANGE_PRECEDING_VEHICLE],tracks,instance,static_info,LC_moment_global,1,features_of_single_LC) #offset
        features_of_single_LC[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_LC[IS_LANE_TO_CHANGE_ALONGSIDE_VEHICLE],tracks,instance,static_info,LC_moment_global,1,features_of_single_LC) #offset
        features_of_single_LC[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_LC[IS_LANE_TO_CHANGE_FOLLOWING_VEHICLE],tracks,instance,static_info,LC_moment_global,1,features_of_single_LC) #offset
        features_of_single_LC[GAP]=get_gap(LC_moment_global,instance,tracks,features_of_single_LC,Direction)       
        features_of_single_LC[FORWARD_LANE_TO_CHANGE_THW]=get_forward_LTC_thwdhwttc(LC_moment_global,instance, tracks, features_of_single_LC, Direction, 1)#offset
        features_of_single_LC[FORWARD_LANE_TO_CHANGE_DHW]=get_forward_LTC_thwdhwttc(LC_moment_global,instance, tracks, features_of_single_LC, Direction, 2)#offset
        features_of_single_LC[FORWARD_LANE_TO_CHANGE_TTC]=get_forward_LTC_thwdhwttc(LC_moment_global,instance, tracks, features_of_single_LC, Direction, 3)#offset
        features_of_single_LC[BACKWARD_LANE_TO_CHANGE_THW]=get_backward_LTC_thwhdwttc(LC_moment_global,instance, tracks, features_of_single_LC, Direction, 1)#offset
        features_of_single_LC[BACKWARD_LANE_TO_CHANGE_DHW]=get_backward_LTC_thwhdwttc(LC_moment_global,instance, tracks, features_of_single_LC, Direction, 2)#offset
        features_of_single_LC[BACKWARD_LANE_TO_CHANGE_TTC]=get_backward_LTC_thwhdwttc(LC_moment_global,instance, tracks, features_of_single_LC, Direction, 3)#offset
        #LC start and end 
        
        LC_start_localidx,is_complete_start=get_LC_start_moment_local(features_of_single_LC,LC_moment_local)
        LC_end_localidx,is_complete_end=get_LC_end_moment_local(features_of_single_LC,LC_moment_local)
        
        LC_action_start_localidx,is_complete_action_start=get_LC_action_start_local(features_of_single_LC,LC_start_localidx,is_complete_start)
        LC_action_end_localidx,is_complete_action_end=get_LC_action_end_local(features_of_single_LC,LC_end_localidx,is_complete_end)

        LC_start_global_frame=int(instance[FRAME][LC_start_localidx])
        LC_end_global_frame=int(instance[FRAME][LC_end_localidx])
        
        LC_action_start_global_frame=int(instance[FRAME][LC_action_start_localidx])
        LC_action_end_global_frame=int(instance[FRAME][LC_action_end_localidx])        
        
        LC_interaction=get_intention_action_moment_attribute(LC_moment_global,instance,features_of_single_LC,tracks,Direction,
                                                             LC_start_localidx,is_complete_start,LC_start_global_frame,
                                                             LC_action_start_localidx,is_complete_action_start,LC_action_start_global_frame)
        
        features_of_single_LC[LC_INTERACTION]=LC_interaction
        features_of_single_LC[LC_ACTION_START_LOCAL_FRAME]=LC_action_start_localidx
        features_of_single_LC[LC_ACTION_END_LOCAL_FRAME]=LC_action_end_localidx
        features_of_single_LC[LC_ACTION_START_GLOBAL_FRAME]=LC_action_start_global_frame
        features_of_single_LC[LC_ACTION_END_GLOBAL_FRAME]=LC_action_end_global_frame
        
        features_of_single_LC[LC_START_GLOBAL_FRAME]=LC_start_global_frame
        features_of_single_LC[LC_END_GLOBAL_FRAME]=LC_end_global_frame
        features_of_single_LC[LC_START_LOCAL_FRAME]=LC_start_localidx
        features_of_single_LC[LC_END_LOCAL_FRAME]=LC_end_localidx
        
        features_of_single_LC[IS_COMPLETE_START]=is_complete_start
        features_of_single_LC[IS_COMPLETE_END]=is_complete_end
        
        features_of_single_LC[LC_MOMENT_LOCAL]=LC_moment_local
        features_of_single_LC[LC_MOMENT_GLOBAL]=LC_moment_global
        
        return_trajectory[idx+1]=features_of_single_LC
        
        trajectory_count+=1
    
    global UPPER_LANE_WIDTH_DICT
    global LOWER_LANE_WIDTH_DICT   
    
    UPPER_LANE_WIDTH_DICT={}
    LOWER_LANE_WIDTH_DICT={}
    
    return return_trajectory

def extraction_BL_in(tracks, static_info, meta_dict, curid)->dict:
    # Assert
    assert isinstance(tracks, list) and len(tracks) != 0
    assert isinstance(static_info, dict) and len(tracks) == len(static_info)
    assert isinstance(meta_dict, dict)
    
    # Initialization
    return_trajectory = {}
    trajectory_count=0
    generate_vehile_in_each_frame(tracks)   
    
    for idx, instance in enumerate(tracks):
        if idx+1 != curid:
            continue
        
        # Preliminary judgment of borrow lane
        if static_info[idx+1][NUMBER_LANE_CHANGES] != 2:
            continue
        # Basic info: instance, static, meta
        static=static_info[idx+1]
        meta=meta_dict
        
        # Accurate judgment of borrow lane
        total_frame_len=len(instance[FRAME])
        init_lane=instance[LANE_ID][0]
        final_lane=instance[LANE_ID][total_frame_len-1]
        # ---Condition1: If the initial lane and the final lane remain unchanged, it is borrow lane
        if init_lane != final_lane:
            continue
        # It can make more precision results, but more difficulty. Current version is a initial version.
        
        # extract the moment of across lane     
        BL_moment_local=borrow_lane_moment(instance[LANE_ID])#local
        BL_moment_global=BL_moment_local+instance[FRAME][0]-1#global
                
        BL_start_localidx=max(0,BL_moment_local[0]-100)
        BL_middle_localidx=int((BL_moment_local[0]+BL_moment_local[1])/2)
        BL_end_localidx=min(len(instance[LANE_ID])-1,BL_moment_local[1]+100)
        
        BL_start_global_frame=int(instance[FRAME][BL_start_localidx])
        BL_middle_global_frame=int(instance[FRAME][BL_middle_localidx])        
        BL_end_global_frame=int(instance[FRAME][BL_end_localidx])
        
        Direction=int(static[DRIVING_DIRECTION])

        features_of_single_BL={}
        features_of_single_BL[VEHICLE_ID]=np.uint16(idx+1)
        
        #Basic information of dataset (global information)
        features_of_single_BL[DATASET]=0
        features_of_single_BL[SCENE]=np.uint8(0) #0:highway 1:...
        features_of_single_BL[SCENE_AVERAGE_SPEED]=np.float32(scene_average_speed(static_info,static,idx+1))#unit: m/s
        
        #Local environment information of target (ego) vehicle
        features_of_single_BL[CURRENT_LANE_WIDTH]=get_current_lane_width(instance,meta,Direction).astype(np.float32)
        features_of_single_BL[TOTAL_NUM_LANE]=np.uint8(get_total_number_lane(Direction))
        features_of_single_BL[ROAD_SPEED_LIMIT]=np.float32(meta[SPEED_LIMIT]/3.6) if meta[SPEED_LIMIT]!=-1 else np.float32(100)#unit: m/s if no speed limit, default 100
        features_of_single_BL[CURRENT_ROAD_CURVATURE]=np.zeros((total_frame_len,),dtype=np.float32)+0.0001 #default 0.0001
        features_of_single_BL[TRAFFIC_LIGHT]=0
        
        #Driving maneuver classification
        features_of_single_BL[CURRENT_DRIVING_MANEUVER]=np.uint8(1)#0:lane change 1:borrow lane        
        features_of_single_BL[LEFT_RIGHT_MANEUVER]=left_right_maneuver_judgement(instance,Direction,features_of_single_BL[CURRENT_DRIVING_MANEUVER])
        features_of_single_BL[CURRENT_LANE_FEATURE_ID]=get_feature_lane_id(instance,Direction)
        features_of_single_BL[CURRENT_LANE_AVERAGE_SPEED]=get_current_lane_average_speed(tracks,instance,static_info,static,idx+1)
        features_of_single_BL[CURRENT_LANE_SPEED_LIMIT]=get_current_lane_speed_limit(features_of_single_BL[CURRENT_LANE_FEATURE_ID],get_limit_speed(features_of_single_BL[TOTAL_NUM_LANE],meta[SPEED_LIMIT]))
        features_of_single_BL[INTENTION_LANE],BL_LEFT_RIGHT_MANEUVER=get_intention_lane(features_of_single_BL[CURRENT_LANE_FEATURE_ID],BL_middle_localidx,features_of_single_BL[LEFT_RIGHT_MANEUVER])

        # Target vehicle information        
        features_of_single_BL[DISTANCE_TO_LANE]=get_distance_to_lane(instance[BBOX],meta,features_of_single_BL[INTENTION_LANE],Direction,BL_LEFT_RIGHT_MANEUVER)#offset
        features_of_single_BL[CLASS]=1 if static[CLASS] == "Car" else 2
        features_of_single_BL[VEHICLE_WIDTH]=instance[BBOX][0,3].astype(np.float32)
        features_of_single_BL[VEHICLE_HEIGHT]=instance[BBOX][0,2].astype(np.float32)
        features_of_single_BL[S_LOCATION]=get_vehicle_s_location(Direction,instance).astype(np.float32)#offset
        features_of_single_BL[D_LOCATION]=get_vehicle_d_location(Direction, instance).astype(np.float32)#offset
        features_of_single_BL[S_VELOCITY]=get_vehicle_s_velocity(Direction, instance).astype(np.float32)
        features_of_single_BL[D_VELOCITY]=get_vehicle_d_velotity(Direction, instance).astype(np.float32)
        features_of_single_BL[HEADING_ANGLE]=np.arctan(features_of_single_BL[D_VELOCITY]/(features_of_single_BL[S_VELOCITY]+EPSILON)).astype(np.float32) # Heading angle along the current lane 
        features_of_single_BL[S_ACCELERATION]=get_vehicle_s_acceleration(Direction, instance).astype(np.float32)
        features_of_single_BL[D_ACCELERATION]=get_vehicle_d_acceleration(Direction, instance).astype(np.float32)
        features_of_single_BL[YAW_RATE]=np.arctan(features_of_single_BL[D_ACCELERATION]/(features_of_single_BL[S_ACCELERATION]+EPSILON)).astype(np.float32)
        features_of_single_BL[S_JERK]=get_vehicle_jerk(features_of_single_BL[S_ACCELERATION], meta).astype(np.float32)
        features_of_single_BL[D_JERK]=get_vehicle_jerk(features_of_single_BL[D_ACCELERATION], meta).astype(np.float32)
        features_of_single_BL[TTC]=instance[TTC].astype(np.float32)
        features_of_single_BL[THW]=instance[THW].astype(np.float32)
        features_of_single_BL[DHW]=instance[DHW].astype(np.float32)
        features_of_single_BL[LEFT_LANE_TYPE]=get_lane_type(features_of_single_BL[CURRENT_LANE_FEATURE_ID], 0, features_of_single_BL[TOTAL_NUM_LANE])
        features_of_single_BL[RIGHT_LANE_TYPE]=get_lane_type(features_of_single_BL[CURRENT_LANE_FEATURE_ID], 1, features_of_single_BL[TOTAL_NUM_LANE])
        features_of_single_BL[STEERING_RATE_ENTROPY]=get_steering_rate_entropy(features_of_single_BL[HEADING_ANGLE])

        #Interactive information of surrounding vehicles
        features_of_single_BL[IS_PRECEDING_VEHICLE]=instance[PRECEDING_ID].astype(np.uint16)
        features_of_single_BL[IS_FOLLOWING_VEHICLE]=instance[FOLLOWING_ID].astype(np.uint16)
        tmp_LTC_pre=np.zeros((total_frame_len,),dtype=np.uint16)
        tmp_LTC_alo=np.zeros((total_frame_len,),dtype=np.uint16)
        tmp_LTC_fol=np.zeros((total_frame_len,),dtype=np.uint16)
        for index,maneuver in enumerate(BL_LEFT_RIGHT_MANEUVER):
            tmp_LTC_pre[index]=instance[RIGHT_PRECEDING_ID][index].astype(np.uint16) if maneuver else instance[LEFT_PRECEDING_ID][index].astype(np.uint16)
            tmp_LTC_alo[index]=instance[RIGHT_ALONGSIDE_ID][index].astype(np.uint16) if maneuver else instance[LEFT_ALONGSIDE_ID][index].astype(np.uint16)
            tmp_LTC_fol[index]=instance[RIGHT_FOLLOWING_ID][index].astype(np.uint16) if maneuver else instance[LEFT_FOLLOWING_ID][index].astype(np.uint16)            
        features_of_single_BL[IS_LANE_TO_CHANGE_PRECEDING_VEHICLE]=tmp_LTC_pre
        features_of_single_BL[IS_LANE_TO_CHANGE_ALONGSIDE_VEHICLE]=tmp_LTC_alo
        features_of_single_BL[IS_LANE_TO_CHANGE_FOLLOWING_VEHICLE]=tmp_LTC_fol
        features_of_single_BL[PRECEDING_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_BL[IS_PRECEDING_VEHICLE],tracks,instance,static_info,BL_moment_global,0,features_of_single_BL) #offset
        features_of_single_BL[FOLLOWING_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_BL[IS_FOLLOWING_VEHICLE],tracks,instance,static_info,BL_moment_global,0,features_of_single_BL) #offset
        features_of_single_BL[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_BL[IS_LANE_TO_CHANGE_PRECEDING_VEHICLE],tracks,instance,static_info,BL_moment_global,1,features_of_single_BL) #offset
        features_of_single_BL[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_BL[IS_LANE_TO_CHANGE_ALONGSIDE_VEHICLE],tracks,instance,static_info,BL_moment_global,1,features_of_single_BL) #offset
        features_of_single_BL[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE]=get_vehicle_feature(features_of_single_BL[IS_LANE_TO_CHANGE_FOLLOWING_VEHICLE],tracks,instance,static_info,BL_moment_global,1,features_of_single_BL) #offset
        features_of_single_BL[GAP]=get_gap_BL(BL_moment_global,instance,tracks,features_of_single_BL,Direction)       
        features_of_single_BL[FORWARD_LANE_TO_CHANGE_THW]=get_forward_LTC_thwdhwttc_BL(BL_moment_global,instance, tracks, features_of_single_BL, Direction, 1)#offset
        features_of_single_BL[FORWARD_LANE_TO_CHANGE_DHW]=get_forward_LTC_thwdhwttc_BL(BL_moment_global,instance, tracks, features_of_single_BL, Direction, 2)#offset
        features_of_single_BL[FORWARD_LANE_TO_CHANGE_TTC]=get_forward_LTC_thwdhwttc_BL(BL_moment_global,instance, tracks, features_of_single_BL, Direction, 3)#offset
        features_of_single_BL[BACKWARD_LANE_TO_CHANGE_THW]=get_backward_LTC_thwhdwttc_BL(BL_moment_global,instance, tracks, features_of_single_BL, Direction, 1)#offset
        features_of_single_BL[BACKWARD_LANE_TO_CHANGE_DHW]=get_backward_LTC_thwhdwttc_BL(BL_moment_global,instance, tracks, features_of_single_BL, Direction, 2)#offset
        features_of_single_BL[BACKWARD_LANE_TO_CHANGE_TTC]=get_backward_LTC_thwhdwttc_BL(BL_moment_global,instance, tracks, features_of_single_BL, Direction, 3)#offset

        #BL start and end 
        features_of_single_BL[BL_START_GLOBAL_FRAME]=BL_start_global_frame
        features_of_single_BL[BL_MIDDLE_GLOBAL_FRAME]=BL_middle_global_frame
        features_of_single_BL[BL_END_GLOBAL_FRAME]=BL_end_global_frame
        features_of_single_BL[BL_START_LOCAL_FRAME]=BL_start_localidx
        features_of_single_BL[BL_MIDDLE_LOCAL_FRAME]=BL_middle_localidx
        features_of_single_BL[BL_END_LOCAL_FRAME]=BL_end_localidx
        
        return_trajectory[idx+1]=features_of_single_BL
        
        trajectory_count+=1
    
    global UPPER_LANE_WIDTH_DICT
    global LOWER_LANE_WIDTH_DICT   
    
    UPPER_LANE_WIDTH_DICT={}
    LOWER_LANE_WIDTH_DICT={}
    
    return return_trajectory
    
    

def generate_pickle_file(features_LC,save_file):
    import pickle
    with open(save_file,'wb') as fw:
        pickle.dump(features_LC,fw)
       
def generate_csv_file(features_LC,save_file):
    save_mat=None
    file_header=[VEHICLE_ID,DATASET,SCENE,SCENE_AVERAGE_SPEED,CURRENT_LANE_WIDTH,
                 TOTAL_NUM_LANE,ROAD_SPEED_LIMIT,CURRENT_ROAD_CURVATURE,TRAFFIC_LIGHT,CURRENT_DRIVING_MANEUVER,
                 LEFT_RIGHT_MANEUVER,CURRENT_LANE_FEATURE_ID,CURRENT_LANE_AVERAGE_SPEED,CURRENT_LANE_SPEED_LIMIT,INTENTION_LANE,
                 DISTANCE_TO_LANE,CLASS,VEHICLE_WIDTH,VEHICLE_HEIGHT,S_LOCATION,
                 D_LOCATION,S_VELOCITY,D_VELOCITY,HEADING_ANGLE,S_ACCELERATION,
                 D_ACCELERATION,YAW_RATE,S_JERK,D_JERK,DHW,
                 THW,TTC,LEFT_LANE_TYPE,RIGHT_LANE_TYPE,STEERING_RATE_ENTROPY,
                 IS_PRECEDING_VEHICLE,"preceding_class","preceding_width","preceding_height","preceding_s_loc","preceding_d_loc","preceding_s_velo","preceding_d_velo","preceding_s_acc","preceding_d_acc",
                 IS_FOLLOWING_VEHICLE,"following_class","following_width","following_height","following_s_loc","following_d_loc","following_s_velo","following_d_velo","following_s_acc","following_d_acc",
                 IS_LANE_TO_CHANGE_PRECEDING_VEHICLE,"LTC_preceding_class","LTC_preceding_width","LTC_preceding_height","LTC_preceding_s_loc","LTC_preceding_d_loc","LTC_preceding_s_velo","LTC_preceding_d_velo","LTC_preceding_s_acc","LTC_preceding_d_acc",
                 IS_LANE_TO_CHANGE_ALONGSIDE_VEHICLE,"LTC_alongside_class","LTC_alongside_width","LTC_alongside_height","LTC_alongside_s_loc","LTC_alongside_d_loc","LTC_alongside_s_velo","LTC_alongside_d_velo","LTC_alongside_s_acc","LTC_alongside_d_acc",
                 IS_LANE_TO_CHANGE_FOLLOWING_VEHICLE,"LTC_following_class","LTC_following_width","LTC_following_height","LTC_following_s_loc","LTC_following_d_loc","LTC_following_s_velo","LTC_following_d_velo","LTC_following_s_acc","LTC_following_d_acc",
                 GAP,FORWARD_LANE_TO_CHANGE_THW,FORWARD_LANE_TO_CHANGE_DHW,FORWARD_LANE_TO_CHANGE_TTC,
                 BACKWARD_LANE_TO_CHANGE_THW,BACKWARD_LANE_TO_CHANGE_DHW,BACKWARD_LANE_TO_CHANGE_TTC]
    assert len(features_LC) != 0
    for key,value in features_LC.items():
        LC_end_localframe=value[LC_END_LOCAL_FRAME]+1
        LC_start_localframe=value[LC_START_LOCAL_FRAME]
        
        LC_feature_of_each_instance=np.zeros((LC_end_localframe-LC_start_localframe,92),)
        LC_feature_of_each_instance[:,0]+=key
        LC_feature_of_each_instance[:,1]=value[DATASET]
        LC_feature_of_each_instance[:,2]=value[SCENE]
        LC_feature_of_each_instance[:,3]=value[SCENE_AVERAGE_SPEED]
        LC_feature_of_each_instance[:,4]=value[CURRENT_LANE_WIDTH][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,5]=value[TOTAL_NUM_LANE]
        LC_feature_of_each_instance[:,6]=value[ROAD_SPEED_LIMIT]
        LC_feature_of_each_instance[:,7]=value[CURRENT_ROAD_CURVATURE][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,8]=value[TRAFFIC_LIGHT]
        LC_feature_of_each_instance[:,9]=value[CURRENT_DRIVING_MANEUVER]
        LC_feature_of_each_instance[:,10]=value[LEFT_RIGHT_MANEUVER]
        LC_feature_of_each_instance[:,11]=value[CURRENT_LANE_FEATURE_ID][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,12]=value[CURRENT_LANE_AVERAGE_SPEED][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,13]=value[CURRENT_LANE_SPEED_LIMIT][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,14]=value[INTENTION_LANE][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,15]=value[DISTANCE_TO_LANE][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,16]=value[CLASS]
        LC_feature_of_each_instance[:,17]=value[VEHICLE_WIDTH]
        LC_feature_of_each_instance[:,18]=value[VEHICLE_HEIGHT]
        LC_feature_of_each_instance[:,19]=value[S_LOCATION][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,20]=value[D_LOCATION][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,21]=value[S_VELOCITY][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,22]=value[D_VELOCITY][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,23]=value[HEADING_ANGLE][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,24]=value[S_ACCELERATION][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,25]=value[D_ACCELERATION][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,26]=value[YAW_RATE][LC_start_localframe:LC_end_localframe]
        LC_feature_of_each_instance[:,27]=value[S_JERK][LC_start_localframe:LC_end_localframe]        
        LC_feature_of_each_instance[:,28]=value[D_JERK][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,29]=value[DHW][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,30]=value[THW][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,31]=value[TTC][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,32]=value[LEFT_LANE_TYPE][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,33]=value[RIGHT_LANE_TYPE][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,34]=value[STEERING_RATE_ENTROPY]
        
        LC_feature_of_each_instance[:,35]=value[IS_PRECEDING_VEHICLE][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,36]=value[PRECEDING_VEHICLE_FEATURE][CLASS][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,37]=value[PRECEDING_VEHICLE_FEATURE][VEHICLE_WIDTH][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,38]=value[PRECEDING_VEHICLE_FEATURE][VEHICLE_HEIGHT][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,39]=value[PRECEDING_VEHICLE_FEATURE][S_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,40]=value[PRECEDING_VEHICLE_FEATURE][D_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,41]=value[PRECEDING_VEHICLE_FEATURE][S_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,42]=value[PRECEDING_VEHICLE_FEATURE][D_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,43]=value[PRECEDING_VEHICLE_FEATURE][S_ACCELERATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,44]=value[PRECEDING_VEHICLE_FEATURE][D_ACCELERATION][LC_start_localframe:LC_end_localframe] 
 
        LC_feature_of_each_instance[:,45]=value[IS_FOLLOWING_VEHICLE][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,46]=value[FOLLOWING_VEHICLE_FEATURE][CLASS][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,47]=value[FOLLOWING_VEHICLE_FEATURE][VEHICLE_WIDTH][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,48]=value[FOLLOWING_VEHICLE_FEATURE][VEHICLE_HEIGHT][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,49]=value[FOLLOWING_VEHICLE_FEATURE][S_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,50]=value[FOLLOWING_VEHICLE_FEATURE][D_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,51]=value[FOLLOWING_VEHICLE_FEATURE][S_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,52]=value[FOLLOWING_VEHICLE_FEATURE][D_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,53]=value[FOLLOWING_VEHICLE_FEATURE][S_ACCELERATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,54]=value[FOLLOWING_VEHICLE_FEATURE][D_ACCELERATION][LC_start_localframe:LC_end_localframe] 
        
        LC_feature_of_each_instance[:,55]=value[IS_LANE_TO_CHANGE_PRECEDING_VEHICLE][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,56]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][CLASS][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,57]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][VEHICLE_WIDTH][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,58]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][VEHICLE_HEIGHT][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,59]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][S_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,60]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][D_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,61]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][S_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,62]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][D_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,63]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][S_ACCELERATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,64]=value[LANE_TO_CHANGE_PRECEDING_VEHICLE_FEATURE][D_ACCELERATION][LC_start_localframe:LC_end_localframe] 
 
        LC_feature_of_each_instance[:,65]=value[IS_LANE_TO_CHANGE_ALONGSIDE_VEHICLE][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,66]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][CLASS][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,67]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][VEHICLE_WIDTH][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,68]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][VEHICLE_HEIGHT][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,69]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][S_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,70]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][D_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,71]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][S_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,72]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][D_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,73]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][S_ACCELERATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,74]=value[LANE_TO_CHANGE_ALONGSIDE_VEHICLE_FEATURE][D_ACCELERATION][LC_start_localframe:LC_end_localframe] 
   
        LC_feature_of_each_instance[:,75]=value[IS_LANE_TO_CHANGE_FOLLOWING_VEHICLE][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,76]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][CLASS][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,77]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][VEHICLE_WIDTH][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,78]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][VEHICLE_HEIGHT][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,79]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][S_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,80]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][D_LOCATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,81]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][S_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,82]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][D_VELOCITY][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,83]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][S_ACCELERATION][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,84]=value[LANE_TO_CHANGE_FOLLOWING_VEHICLE_FEATURE][D_ACCELERATION][LC_start_localframe:LC_end_localframe] 

        LC_feature_of_each_instance[:,85]=value[GAP][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,86]=value[FORWARD_LANE_TO_CHANGE_THW][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,87]=value[FORWARD_LANE_TO_CHANGE_DHW][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,88]=value[FORWARD_LANE_TO_CHANGE_TTC][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,89]=value[BACKWARD_LANE_TO_CHANGE_THW][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,90]=value[BACKWARD_LANE_TO_CHANGE_DHW][LC_start_localframe:LC_end_localframe] 
        LC_feature_of_each_instance[:,91]=value[BACKWARD_LANE_TO_CHANGE_TTC][LC_start_localframe:LC_end_localframe] 
        
        if save_mat is None:
            save_mat=LC_feature_of_each_instance
        else:
            save_mat=np.concatenate((save_mat,LC_feature_of_each_instance),axis=0)
        
    np.savetxt(save_file,save_mat,delimiter=',')
        
    return 0
