from NGSIM2HighD import NGSIM2HighD
import HighD_Columns as HC 
import NGSIM_Columns as NC 
ngsim_dataset_dir =  "/home/demian/code/dataset/NGSIM/smoothed/"
ngsim_dataset_files = ['trajectories-0400-0415_smoothed_21.csv', 'trajectories-0500-0515_smoothed_21.csv', 'trajectories-0515-0530_smoothed_21.csv']
# 'trajectories-0400-0415.csv', 
#             'trajectories-0500-0515.csv',
#             'trajectories-0515-0530.csv',

converter = NGSIM2HighD(ngsim_dataset_dir, ngsim_dataset_files)
#converter.infer_lane_marking()
converter.convert_tracks_info()
converter.convert_meta_info()
converter.convert_static_info()