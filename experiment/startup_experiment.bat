@echo off
echo "starting up python experiment"

cd "C:\Users\crowding_search_2\Desktop\LocalPC-EXP\Experiment\Crowding_VS_features\experiment"

set /p SJ_NUM=Participant number? (ex: 1):

set /p SES=Session type? (ex: practice, real):

echo "Running participant %SJ_NUM%, session type %SES%"

conda activate "C:\Users\crowding_search_2\crowding"