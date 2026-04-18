# 2026-CAPSTONE

## EMG 

### Reference Data
Data paper: https://www.nature.com/articles/s41597-025-05391-0
Data download: https://springernature.figshare.com/articles/dataset/Comprehensive_Human_Locomotion_and_Electromyography_Dataset_Gait120/27677016
- read_gait_data.m: Process original extracted data from MATLAB file to csv file, output stored in 'output'
- phase1_bronze.py: Process csv data file level 1, output stored in '1-bronze'
- phase2_silver.py: Process csv data file level 2, output stored in '2-silver'
- phase3_gold.py: Process csv data file level 3, output stored in '3-gold'
- phase4_validation.py: Final validation, output stored in '3-gold'
- plot_reference_emg.py: Plot the final processed reference data

For final use, use the file 'reference_emg.csv' in '3-gold'
