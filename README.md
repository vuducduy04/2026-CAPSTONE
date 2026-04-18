# 2026-CAPSTONE

## EMG 

### EMG software
This consists of a complete EMG (Electromyography) signal acquisition and analysis system, including both firmware for data acquisition and a Python application for signal processing and visualization.

#### Structure
- `/firmware` - Arduino-based firmware for EMG signal acquisition using ATmega328P
- `/app` - Python application for EMG signal processing and visualization

#### Prerequisites
- PlatformIO (for firmware development)
- Python 3.12 or higher (for the analysis application)
- Arduino Nano or compatible board

#### Installation
1. **Firmware Setup**
   - Navigate to the `/firmware` directory
   - Use PlatformIO to build and upload the firmware
   - See the firmware README for detailed instructions

2. **Application Setup**
   - Navigate to the `/app` directory
   - Create virture environment -> Install libraries and dependencies
   - Remember to `uv sync` in venv

### Reference Data
**Data pape**r: https://www.nature.com/articles/s41597-025-05391-0


**Data download**: https://springernature.figshare.com/articles/dataset/Comprehensive_Human_Locomotion_and_Electromyography_Dataset_Gait120/27677016

#### Structure:
- `1-bronze`: Output for process level 1
- `2-silver`: Output for process level 2
- `3-gold`: Output for process level 3 and validation
- `output`: Processed data to csv file
- `read_gait_data.m`: Process original extracted data from MATLAB file to csv file
- `phase1_bronze.py`: Process csv data file level 1
- `phase2_silver.py`: Process csv data file level 2
- `phase3_gold.py`: Process csv data file level 3
- `phase4_validation.py`: Final validation
- `plot_reference_emg.py`: Plot the final processed reference data

For final use, use the file `reference_emg.csv` in `3-gold`
