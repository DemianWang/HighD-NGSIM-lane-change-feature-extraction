import os
import pandas
import numpy as np
import HighD_Columns as HC 
import NGSIM_Columns as NC 


class NGSIM2HighD:
    def __init__(self,ngsim_csv_file_dir, files):
        self.ngsim_csv_file_dir = ngsim_csv_file_dir
        self.files = files
        self.ngsim = []
        self.us101 = [
            'trajectories-0750am-0805am.csv',
            'trajectories-0805am-0820am.csv',
            'trajectories-0820am-0835am.csv']
        


    def convert_tracks_info(self):
        """ This method applies following changes:
            1. Delete Unneccessary Coloumns:          
            2. Modify Existing Coloumns:
            3. Compute New Coloumns: 
        """
        for i, traj_file in enumerate(self.files):  
            self.ngsim.append(pandas.read_csv(self.ngsim_csv_file_dir+ traj_file))
            self.ngsim[i] = self.ngsim[i].drop(
                            columns = [
                                NC.GLOBAL_X, 
                                NC.GLOBAL_Y,
                                NC.GLOBAL_TIME,
                                NC.PRECEDING_ID,
                                NC.FOLLOWING_ID,
                                ])
            
            self.ngsim[i] = self.ngsim[i].sort_values([NC.ID, NC.FRAME], ascending = [1, 1])
            ngsim_columns = self.ngsim[i].columns
            ngsim_array = self.ngsim[i].to_numpy()
            NC_dict = {}
            for i,c in enumerate(ngsim_columns):
                NC_dict[c] = i
            

            ngsim_array, SVC_dict = self.transform_frame_features(ngsim_array, NC_dict, us101= (traj_file in self.us101))
            
            highD_columns = [None]* (len(ngsim_columns) + len(SVC_dict))
            # Untransformed Columns
            highD_columns[NC_dict[NC.CLASS]] = NC.CLASS
            highD_columns[NC_dict[NC.VELOCITY]] = NC.VELOCITY # Note: Velocity is changed from feet/s to m/s
            highD_columns[NC_dict[NC.ACCELERATION]] = NC.ACCELERATION # Note: Acceleration is changed from feet/s^2 to m/s^2 
            highD_columns[NC_dict[NC.TOTAL_FRAME]] = NC.TOTAL_FRAME
            
            # Transformed Columns
            highD_columns[NC_dict[NC.ID]] = HC.TRACK_ID
            highD_columns[NC_dict[NC.FRAME]] = HC.FRAME
            highD_columns[NC_dict[NC.X]] = HC.Y # NC.X = HC.Y
            highD_columns[NC_dict[NC.Y]] = HC.X # NC.Y = HC.X
            highD_columns[NC_dict[NC.LENGTH]] = HC.WIDTH # NC.LENGTH = HC.WIDTH
            highD_columns[NC_dict[NC.WIDTH]] = HC.HEIGHT # NC.WIDTH = HC.HEIGHT
            highD_columns[NC_dict[NC.DHW]] = HC.DHW
            highD_columns[NC_dict[NC.THW]] = HC.THW
            highD_columns[NC_dict[NC.LANE_ID]] = HC.LANE_ID
            
            # Added Columns
            highD_columns[len(ngsim_columns) + SVC_dict[HC.PRECEDING_ID]] = HC.PRECEDING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.FOLLOWING_ID]] = HC.FOLLOWING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.LEFT_PRECEDING_ID]] = HC.LEFT_PRECEDING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.LEFT_ALONGSIDE_ID]] = HC.LEFT_ALONGSIDE_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.LEFT_FOLLOWING_ID]] = HC.LEFT_FOLLOWING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.RIGHT_PRECEDING_ID]] = HC.RIGHT_PRECEDING_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.RIGHT_ALONGSIDE_ID]] = HC.RIGHT_ALONGSIDE_ID
            highD_columns[len(ngsim_columns) + SVC_dict[HC.RIGHT_FOLLOWING_ID]] = HC.RIGHT_FOLLOWING_ID
            
            # To dataframe
            transformed_ngsim = pandas.DataFrame(data = ngsim_array, columns = highD_columns)
            transformed_ngsim = transformed_ngsim.sort_values([HC.TRACK_ID, HC.FRAME], ascending=[1,1])
            transformed_ngsim.to_csv(self.ngsim_csv_file_dir + "track_" + traj_file, index=False)

    def correct_vehicle_ids(self, ngsim_data, NC_dict):
        row_num = ngsim_data.shape[0]
        prev_id = -1
        correct_id = 1
        for row_itr in range(row_num):
            current_id = ngsim_data[row_itr, NC_dict[NC.ID]]
            if  current_id != prev_id:
                selected_ind = np.logical_and(ngsim_data[:, NC_dict[NC.ID]]== current_id, np.arange(row_num)>=row_itr)
                ngsim_data[selected_ind, NC_dict[NC.ID]] = correct_id
                prev_id = correct_id
                correct_id +=1
            else:
                continue
        return ngsim_data

    def transform_frame_features(self, ngsim_data, NC_dict, us101, logging = True):
        """
        * Remove merging lanes
        * Remove Motor Cycles
        * Correct Vehicles ID
        * Frames should start from 1
        * Transform from feet to meter.
        * Change X and Y location from front center of vehicle to center. 
        * Extract vehicle IDs of surrounding vehicles.
        
        """
        SVC_dict = {
            HC.PRECEDING_ID:0,
            HC.FOLLOWING_ID:1,
            HC.LEFT_PRECEDING_ID:2,
            HC.LEFT_ALONGSIDE_ID:3,
            HC.LEFT_FOLLOWING_ID:4,
            HC.RIGHT_PRECEDING_ID:5,
            HC.RIGHT_ALONGSIDE_ID:6,
            HC.RIGHT_FOLLOWING_ID:7
        }
        # Remove Motor cycles
        ngsim_data = ngsim_data[ngsim_data[:,NC_dict[NC.CLASS]] != 1,:]
        # The vehicles' IDs are not continuous in NGSIM dataset
        ngsim_data = self.correct_vehicle_ids(ngsim_data, NC_dict)
        
        sorted_ind = np.argsort(ngsim_data[:,NC_dict[NC.FRAME]])
        ngsim_data = ngsim_data[sorted_ind]
        # Remove merging lane
        if us101:
            ngsim_data[ngsim_data[:,NC_dict[NC.LANE_ID]]>6,NC_dict[NC.LANE_ID]] = 6
        
        # Change frame origins to 1
        ngsim_data[:,NC_dict[NC.FRAME]] = ngsim_data[:,NC_dict[NC.FRAME]] - min(ngsim_data[:,NC_dict[NC.FRAME]]) + 1
        # Feet => meter
        ngsim_data[:,NC_dict[NC.X]] = 0.3048 * ngsim_data[:,NC_dict[NC.X]]
        ngsim_data[:,NC_dict[NC.Y]] = 0.3048 * ngsim_data[:,NC_dict[NC.Y]]
        ngsim_data[:,NC_dict[NC.LENGTH]] = 0.3048 * ngsim_data[:,NC_dict[NC.LENGTH]]
        ngsim_data[:,NC_dict[NC.WIDTH]] = 0.3048 * ngsim_data[:,NC_dict[NC.WIDTH]]
        ngsim_data[:,NC_dict[NC.VELOCITY]] = 0.3048 * ngsim_data[:,NC_dict[NC.VELOCITY]]
        ngsim_data[:,NC_dict[NC.ACCELERATION]] = 0.3048 * ngsim_data[:,NC_dict[NC.ACCELERATION]]
        ngsim_data[:,NC_dict[NC.DHW]] = 0.3048 * ngsim_data[:,NC_dict[NC.DHW]]
        # Change Y from front of vehicle to center
        ngsim_data[:,NC_dict[NC.Y]] = ngsim_data[:,NC_dict[NC.Y]] - ngsim_data[:,NC_dict[NC.LENGTH]]/2
        
        augmented_features = np.zeros((ngsim_data.shape[0], 8))
        all_frames = sorted(list(set(ngsim_data[:,NC_dict[NC.FRAME]])))
        max_itr = len(all_frames)
        for itr, frame in enumerate(all_frames):
            if logging and itr%100 == 0:
                print('Processing: ', itr, 'out_of: ', max_itr)
            
            
            selected_ind = ngsim_data[:,NC_dict[NC.FRAME]] == frame
            cur_data = ngsim_data[selected_ind]
            #print("Current Vehicles:{} at: {}".format(cur_data[:,NC_dict[NC.ID]], frame))
            #exit()
            cur_aug_features = augmented_features[selected_ind]
            num_veh = cur_data.shape[0]
            
            for veh_itr in range(num_veh):
                cur_lane = cur_data[veh_itr, NC_dict[NC.LANE_ID]]
                cur_y = cur_data[veh_itr, NC_dict[NC.Y]]
                cur_length = cur_data[veh_itr, NC_dict[NC.LENGTH]]
                #print("ID: {}, Y: {}, Lane: {}".format(cur_data[veh_itr, NC_dict[NC.ID]], cur_y, cur_lane))
                mask = [True]* num_veh
                mask[veh_itr] = False
                cur_data_minus_ev = cur_data[mask]
                
                cur_lane_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.LANE_ID]] == cur_lane)
                left_lane_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.LANE_ID]] == (cur_lane-1))
                right_lane_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.LANE_ID]] == (cur_lane+1))
                preceding_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.Y]]- cur_data_minus_ev[:,NC_dict[NC.LENGTH]] > cur_y)
                following_sv_ind = (cur_data_minus_ev[:,NC_dict[NC.Y]] < cur_y-cur_length)
                alongside_sv_ind = \
                np.logical_and((cur_data_minus_ev[:,NC_dict[NC.Y]] >= (cur_y-cur_length)),
                 ((cur_data_minus_ev[:,NC_dict[NC.Y]]-cur_data_minus_ev[:,NC_dict[NC.LENGTH]]) <= cur_y))

                #pv_id
                pv_cand_data = cur_data_minus_ev[np.logical_and(preceding_sv_ind, cur_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.PRECEDING_ID]] = \
                pv_cand_data[np.argmin(pv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(preceding_sv_ind, cur_lane_sv_ind)) == True else 0
                
                #fv_id
                fv_cand_data = cur_data_minus_ev[np.logical_and(following_sv_ind, cur_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.FOLLOWING_ID]] = \
                fv_cand_data[np.argmax(fv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(following_sv_ind, cur_lane_sv_ind)) == True else 0

                #rpv_id
                rpv_cand_data = cur_data_minus_ev[np.logical_and(preceding_sv_ind, right_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.RIGHT_PRECEDING_ID]] = \
                rpv_cand_data[np.argmin(rpv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(preceding_sv_ind, right_lane_sv_ind)) == True else 0
                
                #rfv_id
                rfv_cand_data = cur_data_minus_ev[np.logical_and(following_sv_ind, right_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.RIGHT_FOLLOWING_ID]] = \
                rfv_cand_data[np.argmax(rfv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(following_sv_ind, right_lane_sv_ind)) == True else 0

                #lpv_id
                lpv_cand_data = cur_data_minus_ev[np.logical_and(preceding_sv_ind, left_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.LEFT_PRECEDING_ID]] = \
                lpv_cand_data[np.argmin(lpv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(preceding_sv_ind, left_lane_sv_ind)) == True else 0
                
                #lfv_id
                lfv_cand_data = cur_data_minus_ev[np.logical_and(following_sv_ind, left_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.LEFT_FOLLOWING_ID]] = \
                lfv_cand_data[np.argmax(lfv_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]]\
                if np.any(np.logical_and(following_sv_ind, left_lane_sv_ind)) == True else 0
                
                #rav_id
                rav_cand_data = cur_data_minus_ev[np.logical_and(alongside_sv_ind, right_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.RIGHT_ALONGSIDE_ID]] = \
                rav_cand_data[np.argmax(rav_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(alongside_sv_ind, right_lane_sv_ind)) == True else 0
                
                #lav_id
                lav_cand_data = cur_data_minus_ev[np.logical_and(alongside_sv_ind, left_lane_sv_ind)]
                cur_aug_features[veh_itr,SVC_dict[HC.LEFT_ALONGSIDE_ID]] = \
                lav_cand_data[np.argmax(lav_cand_data[:,NC_dict[NC.Y]]),NC_dict[NC.ID]] \
                if np.any(np.logical_and(alongside_sv_ind, left_lane_sv_ind)) == True else 0
                #print('SVs: ',cur_aug_features[veh_itr])

            augmented_features[selected_ind] = cur_aug_features

        ngsim_data = np.concatenate((ngsim_data, augmented_features), axis = 1)
        
        
        return ngsim_data, SVC_dict

    
    def convert_static_info(self):
        # TODO:  Export following meta features from NGSIM:
        #  TRAVELED_DISTANCE, MIN_X_VELOCITY, MAX_X_VELOCITY, MEAN_X_VELOCITY, MIN_DHW, MIN_THW, MIN_TTC, NUMBER_LANE_CHANGES
        for i,traj_file in enumerate(self.files):
            ngsim_transformed = pandas.read_csv(self.ngsim_csv_file_dir + "track_" + traj_file)
            static_columns = [HC.TRACK_ID, HC.INITIAL_FRAME, HC.FINAL_FRAME, HC.NUM_FRAMES, HC.DRIVING_DIRECTION]
            ngsim_transformed = ngsim_transformed.sort_values(by=[HC.TRACK_ID])
            track_id_list = ngsim_transformed[HC.TRACK_ID].unique()
            ngsim_columns = ngsim_transformed.columns
            ngsim_array = ngsim_transformed.to_numpy()
            HC_dict = {}
            for i,c in enumerate(ngsim_columns):
                HC_dict[c] = i
            static_data = np.zeros((len(track_id_list), len(static_columns)))
            
            for itr, track_id in enumerate(track_id_list):
                cur_track_data = ngsim_array[ngsim_array[:, HC_dict[HC.TRACK_ID]]==track_id] 
                initial_frame = min(cur_track_data[:,HC_dict[HC.FRAME]])
                final_frame = max(cur_track_data[:,HC_dict[HC.FRAME]])
                num_frame = final_frame - initial_frame
                driving_dir = 2
                static_data[itr,:] = [track_id, initial_frame, final_frame, num_frame, driving_dir]

            static = pandas.DataFrame(data = static_data, columns = static_columns)
            static.to_csv(self.ngsim_csv_file_dir + 'static_'+traj_file, index = False)
    
    def get_range(self, columns):
        for ind,traj_file in enumerate(self.files):
            ngsim_transformed = pandas.read_csv(self.ngsim_csv_file_dir + "track_" + traj_file)
            for column in columns:
                print("{}=> Min: {}, Max:{}".format(column, ngsim_transformed[column].min(), ngsim_transformed[column].max()))

            
    def convert_meta_info(self):
        # TODO: Export following meta features from NGSIM:
        #  SPEED_LIMIT, MONTH, WEEKDAY, START_TIME, DURATION, TOTAL_DRIVEN_DISTANCE, TOTAL_DRIVEN_TIME, N_CARS, N_TRUCKS
        for ind,traj_file in enumerate(self.files):
            ngsim_transformed = pandas.read_csv(self.ngsim_csv_file_dir + "track_" + traj_file)
            meta_columns = [HC.ID, HC.FRAME_RATE, HC.LOCATION_ID, HC.N_VEHICLES, HC.UPPER_LANE_MARKINGS, HC.LOWER_LANE_MARKINGS]
            ngsim_transformed = ngsim_transformed.sort_values(by=[HC.LANE_ID])
            max_lane = int(ngsim_transformed[HC.LANE_ID].max())
            ngsim_columns = ngsim_transformed.columns
            ngsim_array = ngsim_transformed.to_numpy()
            HC_dict = {}
            for i,c in enumerate(ngsim_columns):
                HC_dict[c] = i
            
            lower_lanes = np.zeros((max_lane+2))
            average_y = np.zeros((max_lane))
            min_y = np.zeros((max_lane))
            max_y = np.zeros((max_lane))
            for lane in range(max_lane):
                lane_id = lane+1
                average_y[lane] = np.mean(ngsim_array[ngsim_array[:,HC_dict[HC.LANE_ID]] == lane_id, HC_dict[HC.Y]])
                min_y[lane] = min(ngsim_array[ngsim_array[:,HC_dict[HC.LANE_ID]] == lane_id, HC_dict[HC.Y]])
                max_y[lane] = max(ngsim_array[ngsim_array[:,HC_dict[HC.LANE_ID]] == lane_id, HC_dict[HC.Y]])
            print("min y: {}".format(min_y))
            print("max y: {}".format(max_y))     
            for lane in range(max_lane+1):
                lane_id = lane+1
                if lane_id ==1 or lane_id == max_lane+1:
                    continue
                lower_lanes[lane] = average_y[lane-1] + (average_y[lane] - average_y[lane-1])/2
            lower_lanes[0] = lower_lanes[1] - 2*(lower_lanes[1] - average_y[0])
            lower_lanes[-2] = lower_lanes[-3] + 2*(average_y[-1] - lower_lanes[-3])
            lower_lanes[-1] = lower_lanes[-2] + 3*(average_y[-1] - lower_lanes[-3])
            
            upper_lanes = np.array([lower_lanes[-1], lower_lanes[-1]])
            print("Estimated Lower Lane Markings: {}".format(lower_lanes))
            # Note: Upper lanes are not recorded in NGSIM, we arbitrary set some values to them.
            meta_data = np.array([ind, 10, ind, len(ngsim_transformed[HC.TRACK_ID].unique()), 0, 0])
            print(meta_data)
            meta = pandas.DataFrame(data = [meta_data], columns = meta_columns)
            meta = meta.astype(object)
            meta.iloc[0,-2] = ';'.join([str(lane_mark) for lane_mark in upper_lanes])
            meta.iloc[0,-1] = ';'.join([str(lane_mark) for lane_mark in lower_lanes])
            meta.to_csv(self.ngsim_csv_file_dir + 'meta_'+traj_file, index = False)
        
    def infer_lane_marking(self):
        for i,traj_file in enumerate(self.files):
            meta = pandas.read_csv(self.ngsim_csv_file_dir + 'meta_'+traj_file)
            ngsim_transformed = pandas.read_csv(self.ngsim_csv_file_dir + "track_" + traj_file)
            ngsim_transformed = ngsim_transformed.sort_values(by=[HC.TRACK_ID, HC.FRAME])
            max_lane = int(ngsim_transformed[HC.LANE_ID].max())
            lane_locs = [[] for i in range(max_lane-1)]
            track_id_list = ngsim_transformed[HC.TRACK_ID].unique()
            ngsim_columns = ngsim_transformed.columns
            ngsim_array = ngsim_transformed.to_numpy()
            HC_dict = {}
            for i,c in enumerate(ngsim_columns):
                HC_dict[c] = i
            for itr, track_id in enumerate(track_id_list):
                cur_track_data = ngsim_array[ngsim_array[:, HC_dict[HC.TRACK_ID]]==track_id] 
                initial_lane = cur_track_data[0, HC_dict[HC.LANE_ID]]
                new_lane_ind = np.ones_like(cur_track_data[:, HC_dict[HC.LANE_ID]], dtype=np.bool)
                print(track_id)
                for i in range(len(cur_track_data[:, HC_dict[HC.LANE_ID]])):
                    print("{}:{}:{}".format(i,cur_track_data[i, HC_dict[HC.LANE_ID]],cur_track_data[i, HC_dict[HC.Y]]))
                #print("Lane: {}".format(cur_track_data[:, HC_dict[HC.LANE_ID]]))
                #print("Vehicle Lateral position: {}".format(cur_track_data[:, HC_dict[HC.Y]]))
                if track_id == 320:
                    a = 2
                while np.any(np.logical_and(np.not_equal(cur_track_data[:, HC_dict[HC.LANE_ID]], initial_lane), new_lane_ind)):
                    cur_lane = initial_lane
                    lc_ind = np.nonzero(np.logical_and(np.not_equal(cur_track_data[:, HC_dict[HC.LANE_ID]], initial_lane), new_lane_ind))[0][0]
                    new_lane_ind = np.arange(cur_track_data.shape[0])>=lc_ind
                    #np.logical_and(new_lane_ind, (np.not_equal(cur_track_data[:, HC_dict[HC.LANE_ID]], initial_lane)))
                    if cur_lane == 1 and cur_track_data[lc_ind, HC_dict[HC.Y]]>3.7:
                        a =  2
                    if cur_lane == 1 and cur_track_data[lc_ind, HC_dict[HC.Y]]<2.8:
                        a =  2
                    
                    if cur_track_data[lc_ind, HC_dict[HC.LANE_ID]]>cur_lane:
                        lane_locs[int(cur_lane)-1].append(cur_track_data[lc_ind, HC_dict[HC.Y]])
                    else:
                        lane_locs[int(cur_lane)-2].append(cur_track_data[lc_ind, HC_dict[HC.Y]])
                    initial_lane = cur_track_data[lc_ind, HC_dict[HC.LANE_ID]]
            diff =0
            for lane_id, lane_loc in enumerate(lane_locs):
                diff += max(lane_loc)-min(lane_loc)
                print("Lane: {}, Min Y: {}, Max Y: {}, Mean Y: {}, Median Y: {}".format(
                    lane_id+1, min(lane_loc), max(lane_loc), sum(lane_loc)/len(lane_loc), np.median(np.array(lane_loc))
                ))
            diff = diff/len(lane_locs)
            print("Different in lateral position:{}".format(diff))

                    
