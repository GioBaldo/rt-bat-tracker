#!/bin/bash

cd ~/rt-bat-tracker
echo ACTIVATING CONDA ENVIRONMENT
conda init
conda activate BAT
echo RUNNING BAT TRACKER
cd src/rt_bat_tracker
python main.py
#/home/realbat/miniforge3/envs/BAT/bin/python src/rt_bat_tracker/main_other.py