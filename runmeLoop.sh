#If not on Jetson TX2 comment lines 2 and 3
sudo ~/./jetson_clocks.sh
sudo nvpmodel -m0


while true
do
	echo "Press CTRL+C to stop the script execution"
	python3 main_Detect_Track.py
done
