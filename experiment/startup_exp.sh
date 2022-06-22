#!/bin/bash
                
echo "starting up python experiment"

cd C:\Users\crowding_search_2\Desktop\LocalPC-EXP\Experiment\Crowding_VS_features\experiment

conda init bash

conda activate C:\Users\crowding_search_2\crowding 

read -p "Participant number? (ex: 1): " SJ_NUM
read -p "Session type? (ex: practice, real): " SES
echo "Running participant $SJ_NUM, session type $SES"

python main.py $SJ_NUM $SES
