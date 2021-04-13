#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:48:52 2021

@author: demian Wang ## MIS: wangqi123
"""

import os
import sys
import pickle
import argparse
import pickle
from read_origin_dataset import *
from extract_lane_change_feature import extraction_in,generate_csv_file,generate_pickle_file,extraction_BL_in,extraction_trajectory_prediction_feature

def create_args(path,file_ID):
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    parser.add_argument('--input_path',
                        default="{}/data/{}_tracks.csv".format(path,file_ID), 
                        type=str,
                        help='CSV file of the tracks')
    parser.add_argument('--input_static_path', 
                        default="{}/data/{}_tracksMeta.csv".format(path,file_ID),
                        type=str,
                        help='Static meta data file for each track')
    parser.add_argument('--input_meta_path', 
                        default="{}/data/{}_recordingMeta.csv".format(path,file_ID),
                        type=str,
                        help='Static meta data file for the whole video')
    parser.add_argument('--pickle_path', 
                        default="{}/data/{}.pickle".format(path,file_ID), 
                        type=str,
                        help='Converted pickle file that contains corresponding information of the "input_path" file')
    parser.add_argument('--LC_or_BL', 
                        default=1, # LC 1, BC 0 
                        type=int,
                        help='extract lane change feature or borrow lane feature')
    parser.add_argument('--csv_or_pkl', 
                        default=1, # pkl 1, BC 0 
                        type=int,
                        help='save feature as pickle or csv file')
    
    # --- Settings ---
    parser.add_argument('--debug_mode', default=True, type=bool,
                        help='For debug.')

    # --- I/O settings ---
    parser.add_argument('--save_as_pickle', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: you can save the tracks as pickle.')
    
    # --- Finish ---
    parsed_arguments = vars(parser.parse_args())
    return parsed_arguments


if __name__ == '__main__':
    
    Absolute_Path="/home/wqq/dataset/HighD"
    
    for i in range(1,58): 
        print("File {} in processing.".format(i))
        if 0<i and i<10:
            File_Id="0{}".format(i)
        else:
            File_Id="{}".format(i)
        created_arguments = create_args(Absolute_Path,File_Id)
        created_arguments["LC_or_BL"]=1
        LC_BL=[]
        if created_arguments["LC_or_BL"]:
            LC_BL="LC"
        else:
            LC_BL="BL"
        save_file=Absolute_Path+"/save_file/trajpred_data_{}.csv".format(File_Id)
        save_file_pkl=Absolute_Path+"/save_file/trajpred_data_{}.pkl".format(File_Id)
    
        print("Try to find the saved pickle file for better performance.")
        # Read the track csv and convert to useful format
        if os.path.exists(created_arguments["pickle_path"]):
            with open(created_arguments["pickle_path"], "rb") as fp:
                tracks = pickle.load(fp)
            print("Found pickle file {}.".format(created_arguments["pickle_path"]))
        else:
            print("Pickle file not found, csv will be imported now.")
            tracks = read_track_csv(created_arguments)
            print("Finished importing the pickle file.")
    
        if not created_arguments["debug_mode"] and created_arguments["save_as_pickle"] and not os.path.exists(created_arguments["pickle_path"]):
            print("Save tracks to pickle file.")
            with open(created_arguments["pickle_path"], "wb") as fp:
                pickle.dump(tracks, fp)
    
        # Read the static info
        try:
            static_info = read_static_info(created_arguments)
        except:
            print("The static info file is either missing or contains incorrect characters.")
            sys.exit(1)
            
        # Read the video meta
        try:
            meta_dictionary = read_meta_info(created_arguments)
        except:
            print("The video meta file is either missing or contains incorrect characters.")
            sys.exit(1)
       
        if meta_dictionary['locationId']!=1:
            continue
        # Extract the change lane feature    
        try:
            return_final=extraction_trajectory_prediction_feature(tracks, static_info, meta_dictionary)     
            with open(save_file_pkl,'wb') as fw:
                pickle.dump(return_final,fw)
        except:
            print("The feature extraction process meets some errors, please debug it.")
            sys.exit(1)
        
        # try:
        #     if created_arguments["LC_or_BL"]:
        #         features_LC=extraction_in(tracks, static_info, meta_dictionary)      
        #         generate_pickle_file(features_LC,save_file_pkl) if created_arguments["csv_or_pkl"] else generate_csv_file(features_LC,save_file)## generate csv file
        #     else:
        #         features_BL=extraction_BL_in(tracks, static_info, meta_dictionary)
        #         generate_pickle_file(features_BL,save_file_pkl) if created_arguments["csv_or_pkl"] else generate_csv_file(features_BL,save_file)## generate csv file
        # except:
        #     print("The feature extraction process meets some errors, please debug it.")
        #     sys.exit(1)
        
        # datafeature=pickle.load(open(save_file_pkl,'rb'))






