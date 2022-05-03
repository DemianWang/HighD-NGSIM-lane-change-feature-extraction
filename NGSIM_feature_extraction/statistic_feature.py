#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:37:14 2021

@author: demian Wang 
"""
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from scipy import stats
import os
# key
SCENE_AVERAGE_SPEED = "scene_avg_speed"
DHW = "dhw"  # 1
THW = "thw"  # 2
TTC = "ttc"  # 3

FORWARD_LANE_TO_CHANGE_THW = "forward_LTC_thw"  # 4
FORWARD_LANE_TO_CHANGE_DHW = "forward_LTC_dhw"  # 5
FORWARD_LANE_TO_CHANGE_TTC = "forward_LTC_ttc"  # 6
BACKWARD_LANE_TO_CHANGE_THW = "backward_LTC_thw"  # 7
BACKWARD_LANE_TO_CHANGE_DHW = "backward_LTC_dhw"  # 8
BACKWARD_LANE_TO_CHANGE_TTC = "backward_LTC_ttc"  # 9

LC_START_GLOBAL_FRAME = "LC_start_global_frame"
LC_END_GLOBAL_FRAME = "LC_end_global_frame"
LC_START_LOCAL_FRAME = "LC_start_local_frame"
LC_END_LOCAL_FRAME = "LC_end_local_frame"
LC_MOMENT_LOCAL = "LC_moment_local"
LC_MOMENT_GLOBAL = "LC_moment_global"
IS_COMPLETE_START = "is_complete_start"
IS_COMPLETE_END = "is_complete_end"
LC_START_LENGTH = "LC_start_length"
LC_END_LENGTH = "LC_end_length"
LC_TOTAL_LENGTH = "LC_total_length"

COMPLETE_LC = "complete_LC"
COMPLETE_STRAT = "complete_start"
COMPLETE_END = "complete_end"

LC_INTERACTION = "LC_interaction"
LC_ACTION_START_LOCAL_FRAME = "LC_action_start_local_frame"
LC_ACTION_END_LOCAL_FRAME = "LC_action_end_local_frame"
LC_ACTION_START_GLOBAL_FRAME = "LC_action_start_global_frame"
LC_ACTION_END_GLOBAL_FRAME = "LC_action_end_global_frame"

# INTERACTION ATTRIBUTE IN LANE CHANGE OF INTENTION
INTENTION_EGO_PRE_S_DISTANCE = "intention_ego_preceding_S_distance"
INTENTION_EGO_LTCP_S_DISTANCE = "intention_ego_LTCpreceding_S_distance"
INTENTION_EGO_LTCF_S_DISTANCE = "intention_ego_LTCfollowing_S_distance"
INTENTION_GAP = "intention_gap"
INTENTION_EGO_IN_GAP_NORMALIZATION = "intention_ego_in_gap_normalization"
INTENTION_EGO_PRE_S_VELOCITY = "intention_ego_preceding_S_velocity"
INTENTION_EGO_LTCP_S_VELOCITY = "intention_ego_LTCpreceding_S_velocity"
INTENTION_EGO_LTCF_S_VELOCITY = "intention_ego_LTCfollowing_S_velocity"
INTENTION_TTC = "intention_ttc"
INTENTION_LTC_FORWARD_TTC = "intention_LTC_forward_TTC"
INTENTION_LTC_BACKWARD_TTC = "intention_LTC_backward_TTC"
INTENTION_THW = "intention_thw"
INTENTION_LTC_FORWARD_THW = "intention_LTC_forward_THW"
INTENTION_LTC_BACKWARD_THW = "intention_LTC_backward_THW"


# INTERACTION ATTRIBUTE IN LANE CHANGE OF ACTION
ACTION_EGO_PRE_S_DISTANCE = "action_ego_preceding_S_distance"
ACTION_EGO_LTCP_S_DISTANCE = "action_ego_LTCpreceding_S_distance"
ACTION_EGO_LTCF_S_DISTANCE = "action_ego_LTCfollowing_S_distance"
ACTION_GAP = "action_gap"
ACTION_EGO_IN_GAP_NORMALIZATION = "action_ego_in_gap_normalization"
ACTION_EGO_PRE_S_VELOCITY = "action_ego_preceding_S_velocity"
ACTION_EGO_LTCP_S_VELOCITY = "action_ego_LTCpreceding_S_velocity"
ACTION_EGO_LTCF_S_VELOCITY = "action_ego_LTCfollowing_S_velocity"
ACTION_TTC = "action_ttc"
ACTION_LTC_FORWARD_TTC = "action_LTC_forward_TTC"
ACTION_LTC_BACKWARD_TTC = "action_LTC_backward_TTC"
ACTION_THW = "action_thw"
ACTION_LTC_FORWARD_THW = "action_LTC_forward_THW"
ACTION_LTC_BACKWARD_THW = "action_LTC_backward_THW"

# INTERACTION ATTRIBUTE IN LANE CHANGE MOMENT
LCMOMENT_EGO_PRE_S_DISTANCE = "LCmoment_ego_preceding_S_distance"
LCMOMENT_EGO_LTCP_S_DISTANCE = "LCmoment_ego_LTCpreceding_S_distance"
LCMOMENT_EGO_LTCF_S_DISTANCE = "LCmoment_ego_LTCfollowing_S_distance"
LCMOMENT_GAP = "LCmoment_gap"
LCMOMENT_EGO_IN_GAP_NORMALIZATION = "LCmoment_ego_in_gap_normalization"
LCMOMENT_EGO_PRE_S_VELOCITY = "LCmoment_ego_preceding_S_velocity"
LCMOMENT_EGO_LTCP_S_VELOCITY = "LCmoment_ego_LTCpreceding_S_velocity"
LCMOMENT_EGO_LTCF_S_VELOCITY = "LCmoment_ego_LTCfollowing_S_velocity"
LCMOMENT_TTC = "LCmoment_ttc"
LCMOMENT_LTC_FORWARD_TTC = "LCmoment_LTC_forward_TTC"
LCMOMENT_LTC_BACKWARD_TTC = "LCmoment_LTC_backward_TTC"
LCMOMENT_THW = "LCmoment_thw"
LCMOMENT_LTC_FORWARD_THW = "LCmoment_LTC_forward_THW"
LCMOMENT_LTC_BACKWARD_THW = "LCmoment_LTC_backward_THW"

ATTRIBUTE_TOTAL_LEN = "total_len"
ZERO_COUNT = "zero_count"
NEG_COUNT = "neg_count"
BIGGERTHANDEFAULT_COUNT = "bigger_than_default"
TOTAL_MEAN = "total_mean"
TOTAL_VAR = "total_var"
HIGH_ = "high_"
MIDDLE_ = "middle_"
LOW_ = "low_"

# SUB FUNCTION


def create_args(path, file_ID):
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    parser.add_argument('--input_path',
                        default="{}/data/{}_tracks.csv".format(path, file_ID),
                        type=str,
                        help='CSV file of the tracks')
    parser.add_argument('--input_static_path',
                        default="{}/data/{}_tracksMeta.csv".format(
                            path, file_ID),
                        type=str,
                        help='Static meta data file for each track')
    parser.add_argument('--input_meta_path',
                        default="{}/data/{}_recordingMeta.csv".format(
                            path, file_ID),
                        type=str,
                        help='Static meta data file for the whole video')
    parser.add_argument('--pickle_path',
                        default="{}/data/{}.pickle".format(path, file_ID),
                        type=str,
                        help='Converted pickle file that contains corresponding information of the "input_path" file')
    parser.add_argument('--LC_or_BL',
                        default=1,  # LC 1, BC 0
                        type=int,
                        help='extract lane change feature or borrow lane feature')
    parser.add_argument('--csv_or_pkl',
                        default=1,  # pkl 1, BC 0
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


def count_and_remove_zeros(attribute):
    total_len = len(attribute)
    attribute = attribute[attribute[:] != 0]
    return attribute, total_len-len(attribute)


def count_and_remove_negitive(attribute):
    total_len = len(attribute)
    attribute = attribute[attribute[:] > 0]
    return attribute, total_len-len(attribute)


def count_and_remove_biggerthandefault(attribute):  # default 200
    total_len = len(attribute)
    attribute = attribute[attribute[:] < 200]
    return attribute, total_len-len(attribute)


def get_attribute_single(ATTRI, extracted_features):
    attribute = None
    for _, value in extracted_features.items():
        LC_start = value[LC_START_LOCAL_FRAME]
        LC_end = value[LC_END_LOCAL_FRAME]

        attri = value[ATTRI][LC_start:LC_end]
        if attribute is None:
            attribute = attri
        else:
            attribute = np.hstack((attribute, attri))

    return attribute


def get_all_attribute(THWDHWTTC, extracted_features) -> np.ndarray:
    attribute = None
    if THWDHWTTC == 1:
        attribute = get_attribute_single(THW, extracted_features)
    elif THWDHWTTC == 2:
        attribute = get_attribute_single(DHW, extracted_features)
    elif THWDHWTTC == 3:
        attribute = get_attribute_single(TTC, extracted_features)
    elif THWDHWTTC == 4:
        attribute = get_attribute_single(
            FORWARD_LANE_TO_CHANGE_THW, extracted_features)
    elif THWDHWTTC == 5:
        attribute = get_attribute_single(
            FORWARD_LANE_TO_CHANGE_DHW, extracted_features)
    elif THWDHWTTC == 6:
        attribute = get_attribute_single(
            FORWARD_LANE_TO_CHANGE_TTC, extracted_features)
    elif THWDHWTTC == 7:
        attribute = get_attribute_single(
            BACKWARD_LANE_TO_CHANGE_THW, extracted_features)
    elif THWDHWTTC == 8:
        attribute = get_attribute_single(
            BACKWARD_LANE_TO_CHANGE_DHW, extracted_features)
    else:
        attribute = get_attribute_single(
            BACKWARD_LANE_TO_CHANGE_TTC, extracted_features)

    total_len = len(attribute)
    attribute, zeros = count_and_remove_zeros(attribute)
    attribute, negitive = count_and_remove_negitive(attribute)
    attribute, biggerthandefault = count_and_remove_biggerthandefault(
        attribute)
    mean_value = np.mean(attribute)
    var_value = np.var(attribute)

    return attribute, zeros, negitive, biggerthandefault, total_len, mean_value, var_value


def get_different_speed_statistic(HIGHMIDLOW, THWDHWTTC, extracted_features):
    attribute = None
    ATTR = None
    if THWDHWTTC == 1:
        ATTR = THW
    elif THWDHWTTC == 2:
        ATTR = DHW
    elif THWDHWTTC == 3:
        ATTR = TTC
    elif THWDHWTTC == 4:
        ATTR = FORWARD_LANE_TO_CHANGE_THW
    elif THWDHWTTC == 5:
        ATTR = FORWARD_LANE_TO_CHANGE_DHW
    elif THWDHWTTC == 6:
        ATTR = FORWARD_LANE_TO_CHANGE_TTC
    elif THWDHWTTC == 7:
        ATTR = BACKWARD_LANE_TO_CHANGE_THW
    elif THWDHWTTC == 8:
        ATTR = BACKWARD_LANE_TO_CHANGE_DHW
    else:
        ATTR = BACKWARD_LANE_TO_CHANGE_TTC

    if HIGHMIDLOW == 1:  # high
        for _, value in extracted_features.items():
            if not (80/3.6 <= value[SCENE_AVERAGE_SPEED]):
                continue
            LC_start = value[LC_START_LOCAL_FRAME]
            LC_end = value[LC_END_LOCAL_FRAME]

            attri = value[ATTR][LC_start:LC_end]
            if attribute is None:
                attribute = attri
            else:
                attribute = np.hstack((attribute, attri))

    elif HIGHMIDLOW == 2:  # middle
        for _, value in extracted_features.items():
            if not (40/3.6 <= value[SCENE_AVERAGE_SPEED] and value[SCENE_AVERAGE_SPEED] < 80/3.6):
                continue
            LC_start = value[LC_START_LOCAL_FRAME]
            LC_end = value[LC_END_LOCAL_FRAME]

            attri = value[ATTR][LC_start:LC_end]
            if attribute is None:
                attribute = attri
            else:
                attribute = np.hstack((attribute, attri))
    else:
        for _, value in extracted_features.items():
            if not (value[SCENE_AVERAGE_SPEED] <= 40/3.6):
                continue
            LC_start = value[LC_START_LOCAL_FRAME]
            LC_end = value[LC_END_LOCAL_FRAME]

            attri = value[ATTR][LC_start:LC_end]
            if attribute is None:
                attribute = attri
            else:
                attribute = np.hstack((attribute, attri))

    total_len = len(attribute)
    attribute, zeros = count_and_remove_zeros(attribute)
    attribute, negitive = count_and_remove_negitive(attribute)
    attribute, biggerthandefault = count_and_remove_biggerthandefault(
        attribute)
    mean_value = np.mean(attribute)
    var_value = np.var(attribute)

    return attribute, zeros, negitive, biggerthandefault, total_len, mean_value, var_value


def get_all_different_speed(THWDHWTTC, extracted_features):
    attribute = {}

    attri, attribute[ZERO_COUNT], attribute[NEG_COUNT],\
        attribute[BIGGERTHANDEFAULT_COUNT], attribute[ATTRIBUTE_TOTAL_LEN],\
        attribute[TOTAL_MEAN], attribute[TOTAL_VAR] = get_all_attribute(
            THWDHWTTC, extracted_features)

    attri, attribute[HIGH_+ZERO_COUNT], attribute[HIGH_+NEG_COUNT],\
        attribute[HIGH_+BIGGERTHANDEFAULT_COUNT], attribute[HIGH_+ATTRIBUTE_TOTAL_LEN],\
        attribute[HIGH_+TOTAL_MEAN], attribute[HIGH_ +
                                               TOTAL_VAR] = get_different_speed_statistic(1, THWDHWTTC, extracted_features)

    attri, attribute[MIDDLE_+ZERO_COUNT], attribute[MIDDLE_+NEG_COUNT],\
        attribute[MIDDLE_+BIGGERTHANDEFAULT_COUNT], attribute[MIDDLE_+ATTRIBUTE_TOTAL_LEN],\
        attribute[MIDDLE_+TOTAL_MEAN], attribute[MIDDLE_ +
                                                 TOTAL_VAR] = get_different_speed_statistic(2, THWDHWTTC, extracted_features)

    attri, attribute[LOW_+ZERO_COUNT], attribute[LOW_+NEG_COUNT],\
        attribute[LOW_+BIGGERTHANDEFAULT_COUNT], attribute[LOW_+ATTRIBUTE_TOTAL_LEN],\
        attribute[LOW_+TOTAL_MEAN], attribute[LOW_ +
                                              TOTAL_VAR] = get_different_speed_statistic(3, THWDHWTTC, extracted_features)

    attri11 = attri[attri[:] < 100]
    plt.hist(attri11, bins=50, alpha=0.5, rwidth=0.8)
    plt.xlabel('LTC_BACKWARD_TTC')
    plt.ylabel('Number')
    plt.show()

    return attribute, attri


# THWDHWTTC: thw:1 dhw:2 ttc:3
def analysis_thwdhwttc(THWDHWTTC, extracted_features) -> dict:
    attribute = {}
    attribute, attri = get_all_different_speed(THWDHWTTC, extracted_features)

    # attri11=attri[attri[:]<100]
    # plt.hist(attri11,bins=50,alpha=0.5,rwidth=0.8)
    # plt.xlabel('DHW')
    # plt.ylabel('Number')
    # plt.show()

    return attribute


def analysis_LTCthwdhwttc(LTC_THWDHWTTC, extracted_features) -> dict:
    attribute = {}
    attribute, attri = get_all_different_speed(
        LTC_THWDHWTTC, extracted_features)

    attri11 = attri[attri[:] < 100]
    plt.hist(attri11, bins=50, alpha=0.5, rwidth=0.8)
    plt.xlabel('DHW')
    plt.ylabel('Number')
    plt.show()
    return attribute


def start_end_total(value):
    LC_moment_local = value[LC_MOMENT_LOCAL]
    start = None
    end = None
    total = None
    if value[IS_COMPLETE_START]:
        start = (LC_moment_local-value[LC_ACTION_START_LOCAL_FRAME])/25
    if value[IS_COMPLETE_END]:
        end = (value[LC_ACTION_END_LOCAL_FRAME]-LC_moment_local)/25
    if value[IS_COMPLETE_END] and value[IS_COMPLETE_START]:
        total = (value[LC_ACTION_END_LOCAL_FRAME] -
                 value[LC_ACTION_START_LOCAL_FRAME])/25

    return start, end, total


def get_different_speed_LC_time(SPEED, value):
    start = None
    end = None
    total = None
    LC_moment_local = value[LC_MOMENT_LOCAL]
    if SPEED == 1:  # low
        if value[SCENE_AVERAGE_SPEED] <= 40/3.6:
            start, end, total = start_end_total(value)
    elif SPEED == 2:  # mid
        if 40/3.6 <= value[SCENE_AVERAGE_SPEED] and value[SCENE_AVERAGE_SPEED] < 80/3.6:
            start, end, total = start_end_total(value)
    elif SPEED == 3:  # high
        if 80/3.6 <= value[SCENE_AVERAGE_SPEED]:
            start, end, total = start_end_total(value)
    else:
        start, end, total = start_end_total(value)

    return start, end, total


def get_LC_time_statistic(extracted_features, SPEED):
    attribute = {}
    LC_start_time = []
    LC_end_time = []
    LC_total_time = []
    for _, value in extracted_features.items():
        start, end, total = get_different_speed_LC_time(SPEED, value)
        if start:
            LC_start_time.append(start)
        if end:
            LC_end_time.append(end)
        if total:
            LC_total_time.append(total)

    attribute[LC_START_LENGTH] = np.array(LC_start_time)
    attribute[LC_END_LENGTH] = np.array(LC_end_time)
    attribute[LC_TOTAL_LENGTH] = np.array(LC_total_time)

    return attribute


def analysis_LC_time(extracted_features, SPEED) -> dict:
    attribute = {}
    attribute = get_LC_time_statistic(extracted_features, SPEED)

    attri11 = attribute[LC_TOTAL_LENGTH]
    print("mean: ", np.mean(attri11))
    print("var: ", np.var(attri11))
    plt.hist(attri11, bins=36, alpha=0.8, rwidth=0.8)
    plt.xlabel(LC_END_LENGTH+" (unit: s)")
    plt.ylabel('Number')
    plt.show()

    return attribute


ACTION_EGO_PRE_S_DISTANCE = "action_ego_preceding_S_distance"
ACTION_EGO_LTCP_S_DISTANCE = "action_ego_LTCpreceding_S_distance"
ACTION_EGO_LTCF_S_DISTANCE = "action_ego_LTCfollowing_S_distance"
ACTION_GAP = "action_gap"
ACTION_EGO_IN_GAP_NORMALIZATION = "action_ego_in_gap_normalization"
ACTION_EGO_PRE_S_VELOCITY = "action_ego_preceding_S_velocity"
ACTION_EGO_LTCP_S_VELOCITY = "action_ego_LTCpreceding_S_velocity"
ACTION_EGO_LTCF_S_VELOCITY = "action_ego_LTCfollowing_S_velocity"
ACTION_TTC = "action_ttc"
ACTION_LTC_FORWARD_TTC = "action_LTC_forward_TTC"
ACTION_LTC_BACKWARD_TTC = "action_LTC_backward_TTC"
ACTION_THW = "action_thw"
ACTION_LTC_FORWARD_THW = "action_LTC_forward_THW"
ACTION_LTC_BACKWARD_THW = "action_LTC_backward_THW"


def extract_cluster_data(extracted_features):
    cluster_data = None
    ATTR = [LCMOMENT_EGO_LTCP_S_DISTANCE, LCMOMENT_EGO_LTCF_S_DISTANCE, LCMOMENT_LTC_FORWARD_TTC,
            LCMOMENT_LTC_BACKWARD_TTC, LCMOMENT_LTC_FORWARD_THW, LCMOMENT_LTC_BACKWARD_THW,
            LCMOMENT_GAP, LCMOMENT_EGO_IN_GAP_NORMALIZATION]
    for _, value in extracted_features.items():
        attr1 = value[LC_INTERACTION][ACTION_EGO_LTCP_S_DISTANCE]
        attr2 = value[LC_INTERACTION][ACTION_EGO_LTCF_S_DISTANCE]
        if (attr1 or attr1 == 0) and (attr2 or attr2 == 0):
            tmp = None
            for ATTR_ELEMENT in ATTR:
                ele = value[LC_INTERACTION][ATTR_ELEMENT]
                if ATTR_ELEMENT == LCMOMENT_LTC_FORWARD_TTC or ATTR_ELEMENT == LCMOMENT_LTC_BACKWARD_TTC:
                    if ele >= 100:
                        ele = 100
                    elif ele < -100:
                        ele = -100
                if ATTR_ELEMENT == LCMOMENT_LTC_FORWARD_THW or ATTR_ELEMENT == LCMOMENT_LTC_BACKWARD_THW:
                    if ele >= 100:
                        ele = 100
                    elif ele < -100:
                        ele = -100

                if tmp is None:
                    tmp = ele
                else:
                    tmp = np.hstack((tmp, ele))

            if cluster_data is None:
                cluster_data = tmp
            else:
                cluster_data = np.vstack((cluster_data, tmp))

    return cluster_data


def analysis_LC_interaction(extracted_features):
    attribute = None
    ATTR = ACTION_LTC_BACKWARD_TTC

    for _, value in extracted_features.items():
        attri = value[LC_INTERACTION][ATTR]
        if (attri or attri == 0) and \
            (value[LC_INTERACTION][ACTION_EGO_PRE_S_DISTANCE] or value[LC_INTERACTION][ACTION_EGO_PRE_S_DISTANCE] == 0) and \
                (value[LC_INTERACTION][ACTION_EGO_PRE_S_VELOCITY] or value[LC_INTERACTION][ACTION_EGO_PRE_S_VELOCITY] == 0):
            if value[LC_INTERACTION][ACTION_EGO_PRE_S_DISTANCE] <= 50 and 0 <= value[LC_INTERACTION][ACTION_EGO_PRE_S_DISTANCE] and \
                    -1 <= value[LC_INTERACTION][ACTION_EGO_PRE_S_VELOCITY]:
                if attribute is None:
                    attribute = attri
                else:
                    attribute = np.hstack((attribute, attri))

    attribute = attribute[attribute[:] < 20]
    attribute = attribute[0 < attribute[:]]
    print("total len:", len(attribute))
    print("mean: ", np.mean(attribute))
    print("var: ", np.var(attribute))
    print("median: ", np.median(attribute))
    a = stats.mode(attribute)[0][0]
    plt.hist(attribute, bins=40, alpha=0.8, rwidth=0.8)
    plt.xlabel(ATTR)
    plt.ylabel('Number')
    plt.show()

    return 1


def extract_gapnor_data(extracted_features):
    cluster_data = None
    ATTR = [ACTION_EGO_LTCP_S_DISTANCE, ACTION_EGO_LTCF_S_DISTANCE, ACTION_LTC_FORWARD_TTC,
            ACTION_LTC_BACKWARD_TTC, ACTION_LTC_FORWARD_THW, ACTION_LTC_BACKWARD_THW,
            ACTION_EGO_LTCP_S_VELOCITY, ACTION_EGO_LTCF_S_VELOCITY, ACTION_GAP, ACTION_EGO_IN_GAP_NORMALIZATION]
    for _, value in extracted_features.items():
        attr1 = value[LC_INTERACTION][ACTION_EGO_LTCP_S_DISTANCE]
        attr2 = value[LC_INTERACTION][ACTION_EGO_LTCF_S_DISTANCE]
        if (attr1 or attr1 == 0) and (attr2 or attr2 == 0):
            tmp = None
            for ATTR_ELEMENT in ATTR:
                ele = value[LC_INTERACTION][ATTR_ELEMENT]
                if ATTR_ELEMENT == ACTION_LTC_FORWARD_TTC or ATTR_ELEMENT == ACTION_LTC_BACKWARD_TTC:
                    if ele >= 100:
                        ele = 100
                    elif ele < -100:
                        ele = -100
                if ATTR_ELEMENT == ACTION_LTC_FORWARD_THW or ATTR_ELEMENT == ACTION_LTC_BACKWARD_THW:
                    if ele >= 100:
                        ele = 100
                    elif ele < -100:
                        ele = -100

                if tmp is None:
                    tmp = ele
                else:
                    tmp = np.hstack((tmp, ele))

            if cluster_data is None:
                cluster_data = tmp
            else:
                cluster_data = np.vstack((cluster_data, tmp))

    return cluster_data


# read the extracted features

if __name__ == '__main__':

    # Please modify the path!!!!
    path = os.path.dirname(os.path.abspath(__file__))
    Absolute_Path = path + '/NGSIM/smoothed_NGSIM_highDstructure/save_file/'

    extracted_features = {}

    for i in range(0, 6):
        File_Id = "{}".format(i)

        created_arguments = create_args(Absolute_Path, File_Id)
        created_arguments["LC_or_BL"] = 1
        LC_BL = []
        if created_arguments["LC_or_BL"]:
            LC_BL = "LC"
        else:
            LC_BL = "BL"

        save_file_pkl = Absolute_Path + \
            "{}_vehcile_features_{}.pkl".format(LC_BL, File_Id)

        datafeature = pickle.load(open(save_file_pkl, 'rb'))

        for key, value in datafeature.items():
            new_key = File_Id+"-"+str(key)
            extracted_features[new_key] = value

    # analysis the extracted data

    # analysis_thwdhwttc(1,extracted_features)
    # analysis_thwdhwttc(2,extracted_features)
    # analysis_thwdhwttc(3,extracted_features)
    # analysis_LTCthwdhwttc(4, extracted_features)
    # analysis_LTCthwdhwttc(5, extracted_features)
    # analysis_LTCthwdhwttc(6, extracted_features)
    # analysis_LTCthwdhwttc(7, extracted_features)
    # analysis_LTCthwdhwttc(8, extracted_features)
    # analysis_LTCthwdhwttc(9, extracted_features)

    # analysis_LC_time(extracted_features,0)
    # analysis_LC_time(extracted_features,1)
    # analysis_LC_time(extracted_features,2)
    # analysis_LC_time(extracted_features,3)
    cluster_data = extract_gapnor_data(extracted_features)
    # Please modify the path!!!!
    path = os.path.dirname(os.path.abspath(__file__))
    save_file = path+'/NGSIM_feature_extraction/gap_normal_action.pkl'
    with open(save_file, "wb") as fp:
        pickle.dump(cluster_data, fp)

    # analysis_LC_interaction(extracted_features)
