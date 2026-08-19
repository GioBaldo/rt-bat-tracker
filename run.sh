#!/bin/bash

cd ~/rt-bat-tracker
echo ACTIVATING CONDA ENVIRONMENT
conda activate BAT
echo RUNNING BAT TRACKER
cd src/rt_bat_tracker
python main_other.py
#/home/realbat/miniforge3/envs/BAT/bin/python src/rt_bat_tracker/main_other.py