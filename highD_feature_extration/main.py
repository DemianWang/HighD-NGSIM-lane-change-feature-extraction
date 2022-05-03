#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:48:52 2021

@author: demian Wang ## 
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Sequence
from read_origin_dataset import *
from extract_lane_change_feature import extraction_in, generate_csv_file, generate_pickle_file, extraction_BL_in


def create_args():
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    parser.add_argument('--path', type=str,
                        help="Path to hidhD dataset (01_tracks.csv, ...).\
                             PS: 58,59,60 cannot extract features currently, please delete")
    parser.add_argument('--LC_or_BL',
                        default=1,  # LC 1, BL 0
                        type=int,
                        help='extract lane change feature or borrow lane feature')
    parser.add_argument('--csv_or_pkl',
                        default=1,  # pkl 1, BL 0
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


def get_highD_file(root):
    root_dir = Path(root)
    seq_list: Sequence[Path] = [x for x in os.listdir(root_dir)]
    assert len(seq_list) > 0
    file_dict = dict()

    for x in seq_list:
        try:
            curid = int(x[0:2])
        except:
            continue
        if curid not in file_dict.keys():
            file_dict[curid] = dict()

        if x[3:] == "tracks.csv":
            file_dict[curid]['tr'] = x
        elif x[3:] == "tracksMeta.csv":
            file_dict[curid]['trmt'] = x
        elif x[3:] == "highway.jpg":
            file_dict[curid]['jpg'] = x
        elif x[3:] == "recordingMeta.csv":
            file_dict[curid]['rec'] = x
        elif x[3:] == ".pickle":
            file_dict[curid]['pkl'] = x
        else:
            pass

    if len(file_dict) == 0:
        print("No files found, please check your input path.")
        sys.exit(1)

    return file_dict


if __name__ == '__main__':
    created_arguments = create_args()
    LC_BL = "LC" if created_arguments["LC_or_BL"] else "BL"

    file_dict = get_highD_file(created_arguments['path'])
    for key, value in file_dict.items():
        default_path = created_arguments['path']
        print("File {} in processing.".format(key))

        save_file = os.path.join(default_path, "save_file")
        if not os.path.exists(save_file):
            os.makedirs(save_file)

        save_file_path = os.path.join(
            save_file, "{}_vehcile_features_{}.csv".format(LC_BL, key))

        save_file_pkl_path = os.path.join(
            save_file, "{}_vehcile_features_{}.pkl".format(LC_BL, key))

        print("Try to find the saved pickle file for better performance.")
        # Read the track csv and convert to useful format
        if 'pkl' in value and os.path.exists(os.path.join(save_file, value['pkl'])):
            with open(os.path.join(save_file, value['pkl']), "rb") as fp:
                tracks = pickle.load(fp)
            print("Found pickle file {}.".format(
                os.path.join(save_file, value['pkl'])))
        else:
            print("Pickle file not found, csv will be imported now.")
            tracks = read_track_csv(os.path.join(default_path, value['tr']))
            print("Finished importing the pickle file.")

        if not created_arguments["debug_mode"] and created_arguments["save_as_pickle"] and not os.path.exists(created_arguments["pickle_path"]):
            print("Save tracks to pickle file.")
            keystr = (str(key) if key > 10 else '0'+str(key)) + '.pickle'
            pickle_path = os.path.join(default_path, keystr)
            with open(pickle_path, "wb") as fp:
                pickle.dump(tracks, fp)

        # Read the static info
        try:
            static_info = read_static_info(
                os.path.join(default_path, value['trmt']))
        except:
            print(
                "The static info file is either missing or contains incorrect characters.")
            sys.exit(1)

        # Read the video meta
        try:
            meta_dictionary = read_meta_info(
                os.path.join(default_path, value['rec']))
        except:
            print(
                "The video meta file is either missing or contains incorrect characters.")
            sys.exit(1)

        # Extract the change lane feature
        try:
            if created_arguments["LC_or_BL"]:
                features_LC = extraction_in(
                    tracks, static_info, meta_dictionary)
                generate_pickle_file(features_LC, save_file_pkl_path) if created_arguments["csv_or_pkl"] else generate_csv_file(
                    features_LC, save_file_path)  # generate csv file
            else:
                features_BL = extraction_BL_in(
                    tracks, static_info, meta_dictionary)
                generate_pickle_file(features_BL, save_file_pkl_path) if created_arguments["csv_or_pkl"] else generate_csv_file(
                    features_BL, save_file_path)  # generate csv file
        except:
            print("The feature extraction process meets some errors, please debug it.")
            sys.exit(1)

        # datafeature=pickle.load(open(save_file_pkl,'rb'))
