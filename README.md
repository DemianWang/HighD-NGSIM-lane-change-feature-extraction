# HighD-NGSIM-lane-change-feature-extraction
HighD,NGSIM lane change feature extraction and save.  

## HighD dataset
**usage**   
```
cd ./highD_feature_extraction  
python main.py --path YOUR_HIGHD_DATASET_PATH
```
PS: 58,59,60 cannot extract features currently, please delete those files.  

## NGSIM dataset
**usage**   
***Step 1*** Please manually modify the path in ./NGSIM_feature_extraction/NGSIM2HIGHD-master/main.py.

Example:  
```
# Please modify the path!!!!
path = YOUR_DATASET_PATH
Absolute_Path = path + '/NGSIM/smoothed_NGSIM_highDstructure/save_file/'
```

**Then**
```
cd ./NGSIM_feature_extraction/NGSIM2HIGHD-master  
python main.py
```

***Step 2*** Please manually modify the path in ./NGSIM_feature_extraction/main.py.  

**Then**
```
cd ./NGSIM_feature_extraction  
python main.py
```

# Other matters  
The feature extraction code of NGSIM dataset is not well organized.  
But if you have any questions, you can consult me (in issues).  

If you like this repository, please click **star**, thank you~