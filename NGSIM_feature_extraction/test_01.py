#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:51:06 2021

@author: MIS: wangqi123
"""

import matplotlib.pyplot as plt
import pandas as pd


def smooth_dataset(window, train, file_names):
    """
    this function loops over a set of train data, and set of unique vehicle ids
    and for each vehicle id in each training dataset, it requests from helper methods the smoothed
    x, y, vel, accel values and replaces the old values with the smoothed values. Finally the new
    smoothed dataset is printed to a file
    :param dataset:  data frame representing the dataset to smooth it's local X and Y
    :param train: a list of 3 numpy arrays containing the original ngsim data
    """
    # find  unique vehicle ids in all the datasets, in the previous version
    vehicle_ids = [train[0]['Vehicle_ID'].unique(), train[1]['Vehicle_ID'].unique(), train[2]['Vehicle_ID'].unique()]

    # convert to numpy arrays to fascilitate matrix operations to compute velocity and acceleration
    numpy_trains = [train[0].to_numpy(), train[1].to_numpy(), train[2].to_numpy()]

    for i in range(3): #in each dataset
        numpy_train = numpy_trains[i]
        print(f"##### smoothing x, y, vel, accl values in train data {str(i)}")

        # for each unique vehicle id smooth x and y, recompute vel and acel
        for vehicle in vehicle_ids[i]:
            # create a filter for given vehicle id and use it to create a numpy array containing info only for that vehicle
            filter = numpy_train[:,0] == vehicle
            numpy_vehicle_dataset = numpy_train[filter,:]
            
            x=numpy_vehicle_dataset[:,local_x]
            y=numpy_vehicle_dataset[:,local_y]

            plt.plot(y,x)
            plt.show()


def main():
    # smooth window must be an odd value
    smoothing_window = 21
    print(f"Smoothing window is set to {str(smoothing_window)}")

    # change the file names as needed
    global file_names
    file_names = ['trajectories-0750am-0805am.csv', 'trajectories-0805am-0820am.csv', 'trajectories-0820am-0835am.csv']

    # define the index of columns containing vehicle id, time, local x, local y, velocity and acceleration
    # these indexes correspond to the original dataset if not modified
    # the indexes help treat the dataset as matrix and perform smoothing using matrix functions
    global vehicle_id, time_column, local_x, local_y, v_vel, v_acc
    vehicle_id, time_column, local_x, local_y, v_vel, v_acc  = 0, 3, 4, 5, 11, 12

    # specify the path to the input NGSIM dataset and the path to the output smoothed dataset
    global path_to_dataset, path_to_smoothed_dataset
    path_to_dataset = '/home/demian/code/dataset/NGSIM/smoothed/'
    path_to_smoothed_dataset = '/home/demian/code/dataset/NGSIM/smoothed/'

    # load the NGSIM data from the CSV files
    train1 = pd.read_csv(path_to_dataset + file_names[0], engine='c')
    train2 = pd.read_csv(path_to_dataset + file_names[1], engine='c')
    train3 = pd.read_csv(path_to_dataset + file_names[2], engine='c')

    train = [train1, train2, train3]

    smooth_dataset(smoothing_window, train, file_names)

if __name__ == '__main__':
    main()










