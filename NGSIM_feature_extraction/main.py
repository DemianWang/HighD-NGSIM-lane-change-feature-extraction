#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:48:52 2021

@author: demian Wang ## MIS: wangqi123
"""

import os
import sys
import pickle
import pandas
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from extract_lane_change_feature import extraction_in,generate_csv_file,generate_pickle_file,extraction_BL_in

# TRACK FILE
BBOX = "bbox"
FRAMES = "frames"
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "height"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
FRONT_SIGHT_DISTANCE = "frontSightDistance"
BACK_SIGHT_DISTANCE = "backSightDistance"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
PRECEDING_X_VELOCITY = "precedingXVelocity"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"

# STATIC FILE
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
NUM_FRAMES = "numFrames"
CLASS = "class"
DRIVING_DIRECTION = "drivingDirection"
TRAVELED_DISTANCE = "traveledDistance"
MIN_X_VELOCITY = "minXVelocity"
MAX_X_VELOCITY = "maxXVelocity"
MEAN_X_VELOCITY = "meanXVelocity"
MIN_DHW = "minDHW"
MIN_THW = "minTHW"
MIN_TTC = "minTTC"
NUMBER_LANE_CHANGES = "numLaneChanges"

# VIDEO META
ID = "id"
FRAME_RATE = "frameRate"
LOCATION_ID = "locationId"
SPEED_LIMIT = "speedLimit"
MONTH = "month"
WEEKDAY = "weekDay"
START_TIME = "startTime"
DURATION = "duration"
TOTAL_DRIVEN_DISTANCE = "totalDrivenDistance"
TOTAL_DRIVEN_TIME = "totalDrivenTime"
N_VEHICLES = "numVehicles"
N_CARS = "numCars"
N_TRUCKS = "numTrucks"
UPPER_LANE_MARKINGS = "upperLaneMarkings"
LOWER_LANE_MARKINGS = "lowerLaneMarkings"

#NGSIM
v_Vel="v_Vel"
v_Acc="v_Acc"

def read_track_csv(file_name,save_file):
    """
    This method reads the tracks file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the tracks csv file.
    :return: a list containing all tracks as dictionaries.
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(file_name)

    # Use groupby to aggregate track info. Less error prone than iterating over the data.
    grouped = df.groupby([TRACK_ID], sort=False)
    # Efficiently pre-allocate an empty list of sufficient size
    tracks = [None] * grouped.ngroups
    current_track = 0
    for group_id, rows in grouped:
        bounding_boxes = np.transpose(np.array([rows[X].values,
                                                rows[Y].values,
                                                rows[WIDTH].values,
                                                rows[HEIGHT].values]))
        #get X_vel,Y_vel,X_acc,Y_acc  #final tracks is [2:end]
        X_vel=np.zeros((bounding_boxes.shape[0],),dtype=np.float32)
        Y_vel=np.zeros((bounding_boxes.shape[0],),dtype=np.float32)
        for idx in range(1,bounding_boxes.shape[0]):
            # if idx == 231:
            #     aa=1
            X_vel[idx]=(bounding_boxes[idx,0]-bounding_boxes[idx-1,0])*10
            Y_vel[idx]=(bounding_boxes[idx,1]-bounding_boxes[idx-1,1])*10
            flag=-1 if Y_vel[idx]<0 else 1
            
            if np.abs(rows[v_Vel].values[idx])<np.abs(X_vel[idx]):
                Y_vel[idx]=Y_vel[idx]/2

            Y_vel[idx]=(np.sqrt(np.abs(rows[v_Vel].values[idx]*rows[v_Vel].values[idx]-X_vel[idx]*X_vel[idx]))*flag+Y_vel[idx])/2
            
            flag=-1 if X_vel[idx]<0 else 1
            X_vel[idx]=(np.sqrt(np.abs(rows[v_Vel].values[idx]*rows[v_Vel].values[idx]-Y_vel[idx]*Y_vel[idx]))*flag+X_vel[idx])/2

        X_vel=signal.savgol_filter(X_vel[2:], 21, 2)
        Y_vel=signal.savgol_filter(Y_vel[2:], 21, 2)
        
        X_Acc=np.zeros((bounding_boxes.shape[0],),dtype=np.float32)
        Y_Acc=np.zeros((bounding_boxes.shape[0],),dtype=np.float32)
        for idx in range(2,bounding_boxes.shape[0]):           
            X_Acc[idx]=(bounding_boxes[idx,0]+bounding_boxes[idx-2,0]-2*bounding_boxes[idx-1,0])*100
            Y_Acc[idx]=(bounding_boxes[idx,1]+bounding_boxes[idx-2,1]-2*bounding_boxes[idx-1,1])*100
            flag=-1 if Y_Acc[idx]<0 else 1
            if np.abs(rows[v_Acc].values[idx])<np.abs(X_Acc[idx]):
                Y_Acc[idx]=Y_Acc[idx]/2
            Y_Acc[idx]=(np.sqrt(np.abs(rows[v_Acc].values[idx]*rows[v_Acc].values[idx]-X_Acc[idx]*X_Acc[idx]))*flag+Y_Acc[idx])/2

            flag=-1 if X_Acc[idx]<0 else 1
            X_Acc[idx]=(np.sqrt(np.abs(rows[v_Acc].values[idx]*rows[v_Acc].values[idx]-Y_Acc[idx]*Y_Acc[idx]))*flag+X_Acc[idx])/2
 
        X_Acc=signal.savgol_filter(X_Acc[2:], 21, 2)
        Y_Acc=signal.savgol_filter(Y_Acc[2:], 21, 2)            
            
        tracks[current_track] = {TRACK_ID: np.int64(group_id),  # for compatibility, int would be more space efficient
                                 FRAME: rows[FRAME].values[2:],
                                 BBOX: bounding_boxes[2:,:],
                                 X_VELOCITY: X_vel,
                                 Y_VELOCITY: Y_vel,
                                 X_ACCELERATION: X_Acc,
                                 Y_ACCELERATION: Y_Acc,
                                 # FRONT_SIGHT_DISTANCE: rows[FRONT_SIGHT_DISTANCE].values,
                                 # BACK_SIGHT_DISTANCE: rows[BACK_SIGHT_DISTANCE].values,
                                 # THW: rows[THW].values,
                                 # TTC: rows[TTC].values,
                                 # DHW: rows[DHW].values,
                                 # PRECEDING_X_VELOCITY: rows[PRECEDING_X_VELOCITY].values[2:],
                                 CLASS: rows["v_Class"].values[2:],
                                 PRECEDING_ID: rows[PRECEDING_ID].values[2:],
                                 FOLLOWING_ID: rows[FOLLOWING_ID].values[2:],
                                 LEFT_FOLLOWING_ID: rows[LEFT_FOLLOWING_ID].values[2:],
                                 LEFT_ALONGSIDE_ID: rows[LEFT_ALONGSIDE_ID].values[2:],
                                 LEFT_PRECEDING_ID: rows[LEFT_PRECEDING_ID].values[2:],
                                 RIGHT_FOLLOWING_ID: rows[RIGHT_FOLLOWING_ID].values[2:],
                                 RIGHT_ALONGSIDE_ID: rows[RIGHT_ALONGSIDE_ID].values[2:],
                                 RIGHT_PRECEDING_ID: rows[RIGHT_PRECEDING_ID].values[2:],
                                 LANE_ID: rows[LANE_ID].values[2:]
                                 }
        current_track = current_track + 1
        
    print("Save tracks to pickle file.")    
    with open(save_file, "wb") as fp:
        pickle.dump(tracks, fp)
            
    return tracks

def read_static_info(file_name,save_file,tracks):
    """
    This method reads the static info file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(file_name)

    # Declare and initialize the static_dictionary
    static_dictionary = {}
    # lane_change=np.zeros((100,),dtype=np.int64)
    # Iterate over all rows of the csv because we need to create the bounding boxes for each row
    for i_row in range(df.shape[0]):
        track_id = int(df[TRACK_ID][i_row])
        
        instance = tracks[i_row]
        init_lane = instance[LANE_ID][0]
        count=0
        for i in range(1,len(instance[LANE_ID])):
            if instance[LANE_ID][i]!=init_lane:
                count+=1
                init_lane=instance[LANE_ID][i]
        # lane_change[count]+=1
        
        static_dictionary[track_id] = {TRACK_ID: track_id,
                                       # WIDTH: int(df[WIDTH][i_row]),
                                       # HEIGHT: int(df[HEIGHT][i_row]),
                                       INITIAL_FRAME: int(df[INITIAL_FRAME][i_row]),
                                       FINAL_FRAME: int(df[FINAL_FRAME][i_row]),
                                       NUM_FRAMES: int(df[NUM_FRAMES][i_row]),
                                       # CLASS: str(df[CLASS][i_row]),
                                        # DRIVING_DIRECTION: float(df[DRIVING_DIRECTION][i_row]),
                                       # TRAVELED_DISTANCE: float(df[TRAVELED_DISTANCE][i_row]),
                                       # MIN_X_VELOCITY: float(df[MIN_X_VELOCITY][i_row]),
                                       # MAX_X_VELOCITY: float(df[MAX_X_VELOCITY][i_row]),
                                       # MEAN_X_VELOCITY: float(df[MEAN_X_VELOCITY][i_row]),
                                       # MIN_TTC: float(df[MIN_TTC][i_row]),
                                       # MIN_THW: float(df[MIN_THW][i_row]),
                                       # MIN_DHW: float(df[MIN_DHW][i_row]),
                                       NUMBER_LANE_CHANGES: int(count)
                                       }
    print("Save tracks to pickle file.")    
    with open(save_file, "wb") as fp:
        pickle.dump(static_dictionary, fp)
    return static_dictionary

def read_meta_info(file_name):
    """
    This method reads the video meta file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the video meta csv file.
    :return: the meta dictionary containing the general information of the video
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(file_name)

    # Declare and initialize the extracted_meta_dictionary
    extracted_meta_dictionary = {ID: int(df[ID][0]),
                                 FRAME_RATE: int(df[FRAME_RATE][0]),
                                 LOCATION_ID: int(df[LOCATION_ID][0]),
                                 # SPEED_LIMIT: float(df[SPEED_LIMIT][0]),
                                 # MONTH: str(df[MONTH][0]),
                                 # WEEKDAY: str(df[WEEKDAY][0]),
                                 # START_TIME: str(df[START_TIME][0]),
                                 # DURATION: float(df[DURATION][0]),
                                 # TOTAL_DRIVEN_DISTANCE: float(df[TOTAL_DRIVEN_DISTANCE][0]),
                                 # TOTAL_DRIVEN_TIME: float(df[TOTAL_DRIVEN_TIME][0]),
                                 # N_VEHICLES: int(df[N_VEHICLES][0]),
                                 # N_CARS: int(df[N_CARS][0]),
                                 # N_TRUCKS: int(df[N_TRUCKS][0]),
                                 # UPPER_LANE_MARKINGS: np.fromstring(df[UPPER_LANE_MARKINGS][0], sep=";"),
                                 LOWER_LANE_MARKINGS: np.fromstring(df[LOWER_LANE_MARKINGS][0], sep=";")}
    return extracted_meta_dictionary

if __name__ == '__main__':
    
    Absolute_Path="/home/demian/code/dataset/NGSIM/smoothed_NGSIM_highDstructure/"
    
    # file_name=["trajectories-0400-0415_smoothed_21","trajectories-0500-0515_smoothed_21","trajectories-0515-0530_smoothed_21"]
    file_name=["trajectories-0750am-0805am","trajectories-0805am-0820am","trajectories-0820am-0835am",
               "trajectories-0400-0415_smoothed_21","trajectories-0500-0515_smoothed_21","trajectories-0515-0530_smoothed_21"]
    class_name=["track_","meta_","static_"]
    
    
    for i in range(6): 
        print("File {} in processing.".format(i))

        # save_file=Absolute_Path+"/save_file/{}_vehcile_features_{}.csv".format(LC_BL,File_Id)
        # save_file_pkl=Absolute_Path+"save_file/LC_vehcile_features_{}.pkl".format(i+3)
        save_file_pkl=Absolute_Path+"save_file/LC_vehcile_features_{}.pkl".format(i)
        # Read the tracks
        if os.path.exists(Absolute_Path+class_name[0]+file_name[i]+".pkl"):
            with open(Absolute_Path+class_name[0]+file_name[i]+".pkl", "rb") as fp:
                tracks = pickle.load(fp)
        else:
            try:
                tracks = read_track_csv(Absolute_Path+class_name[0]+file_name[i]+".csv",Absolute_Path+class_name[0]+file_name[i]+".pkl")
            except:
                print("The static info file is either missing or contains incorrect characters.")
                sys.exit(1) 

        # Read the static info
        if os.path.exists(Absolute_Path+class_name[2]+file_name[i]+".pkl"):
            with open(Absolute_Path+class_name[2]+file_name[i]+".pkl", "rb") as fp:
                static_info = pickle.load(fp)
        else:
            try:
                static_info = read_static_info(Absolute_Path+class_name[2]+file_name[i]+".csv",Absolute_Path+class_name[2]+file_name[i]+".pkl",tracks)
            except:
                print("The static info file is either missing or contains incorrect characters.")
                sys.exit(1)
            
        # # Read the video meta
        try:
            meta_dictionary = read_meta_info(Absolute_Path+class_name[1]+file_name[i]+".csv")
        except:
            print("The video meta file is either missing or contains incorrect characters.")
            sys.exit(1)
            
        # # Extract the change lane feature    
        try:
            features_LC=extraction_in(tracks, static_info, meta_dictionary)      
            generate_pickle_file(features_LC,save_file_pkl) #if created_arguments["csv_or_pkl"] else generate_csv_file(features_LC,save_file)## generate csv file
            # else:
            #     features_BL=extraction_BL_in(tracks, static_info, meta_dictionary)
            #     generate_pickle_file(features_BL,save_file_pkl) if created_arguments["csv_or_pkl"] else generate_csv_file(features_BL,save_file)## generate csv file
        except:
            print("The feature extraction process meets some errors, please debug it.")
            sys.exit(1)
        
        # datafeature=pickle.load(open(save_file_pkl,'rb'))






