# FPS = 10
ID = "Vehicle_ID"
FRAME = "Frame_ID"
TOTAL_FRAME = "Total_Frames"
GLOBAL_TIME = "Global_Time" # milisecond
# Vehicles travel from top of the image to the bottom
# X in ngsim = Y in HighD, Width in ngsim = hight in highd
X = "Local_X" # front center of vehicle.feet w.r.t left edge of image
Y = "Local_Y" # front center of vehicle,feet w.r.t. highway entry
GLOBAL_X = "Global_X" # feet
GLOBAL_Y = "Global_Y" # feet
LENGTH = "v_Length" # feet # equal to highd WIDTH
WIDTH = "v_Width" #feet # equal to highd Height
CLASS = "v_Class" #1: motorcycle, 2: auto, 3: truck 
VELOCITY = "v_Vel" #feet/s
ACCELERATION = "v_Acc" # feet/s^2
LANE_ID = "Lane_ID"# 1: farthest left (High speed), 5: farthest right(Low Speed)  
PRECEDING_ID = "Preceeding"# 0 means no vehicle
FOLLOWING_ID = "Following" # 0 means no vehicle
LOCATION = "Location" 
O_ZONE = "O_Zone"
D_ZONE = "D_Zone"
INT_ID = "Int_ID"
SECTION_ID = "Section_ID"
DIRECTION = "Direction"
MOVEMENT = "Movement"
DHW = "Space_Hdwy" # Distance from the front-centere of a vehicle to the fron-center of the preceding, feet
THW = "Time_Hdwy" # A value of 9999.99 means vehicle speed is zero