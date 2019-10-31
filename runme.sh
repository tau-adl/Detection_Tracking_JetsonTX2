chmod a+x  ~/Documents/MSC_Project/Current/TrackAndDetect_Yon_Commented
#If not on Jetson TX2 comment lines 2 and 3
#sudo ./jetson_clocks.sh
#sudo nvpmodel -m0

python3 main_Detect_Track.py
