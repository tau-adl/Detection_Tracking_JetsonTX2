from DetectAndTrack import *
import argparse
import setproctitle
setproctitle.setproctitle('Detection_And_Tracking')

### Input
VideoInput = 'Input/Expo884FPS20.mp4'  #Input video
# VideoInput = 'Input/Trim840.mp4'  #Input video

### Output
output = "output"                       #Output folder

### Parameters
conf_thres = 0.6                        # Confidence Threshold for YOLO detection
nms_thres = 0.15                        # Non-Maximal Suppression for YOLO detection=> choses heighst probability detections
doubleCheckThres = 0.90                 # If below thresh. => Double check if detection was accurate in next frame.
trackingConfidenceThres = 0.3           # Threshold for continuing tracking
VerificationInterval = 5 #sec        # Verification time interval in seconds of the tracking

### Flags
Initial_State = "Detection"
show_output = True                      # True: plots output
save_output = False                     # True: saves output in output folder
saveSmoothVideo = True                  # True if you want to complete missing frames by duplicating frame => smooth output video
sliding_Detection = True                # True: uses scanning of the screen with sliding window for enhanced detection
RealTimeSimu = False                      # True: Loads frames according to computation time of itterations
ImportType = 'Queue'                    # Input video import type"JetsonCam", "Queue", "Pre-load", None(= cv2 VideoCapture)

### Models
cfg = 'yolo/model/yolo-drone.cfg'
data = 'yolo/model/yolo-drone.data'
weights = 'yolo/model/yolo-drone_120000_49.weights'
prototxt = 'goturn/nets/tracker.prototxt'
model = 'goturn/nets/tracker.caffemodel'

### Arg parser
ap = argparse.ArgumentParser()
ap.add_argument("--VideoInput", required=False, default=VideoInput, help="Path to input video")
ap.add_argument("--Initial_State", required=False, default=Initial_State, help="Initial state => 'Tracking' or 'Detection'")

ap.add_argument("--conf_thres", required=False, default=conf_thres, help="YOLO detection Confidence Threshold")
ap.add_argument("--nms_thres", required=False, default=nms_thres, help="YOLO detection Non-Maximal Suppression Threshold")
ap.add_argument("--doubleCheckThres", required=False, default=doubleCheckThres, help="Double check thershold")
ap.add_argument("--trackingConfidenceThres", required=False, default=trackingConfidenceThres, help="Tracking Confidence Threshold")
ap.add_argument("--VerificationInterval", required=False, default=VerificationInterval, help="Tracking verification time interval in seconds")

ap.add_argument("--show_output", required=False, default=show_output, help="Show output")
ap.add_argument("--save_output", required=False, default=save_output, help="Save output")
ap.add_argument("--RealTimeSimu", required=False, default=RealTimeSimu, help="Real time simulation")
ap.add_argument("--ImportType", required=False, default=ImportType,
                help="Input video import type.'JetsonCam', 'Queue', 'Pre-load', None(= cv2 VideoCapture)")

args = vars(ap.parse_args())


### INITIALIZATION
detectorAndTracker = DetectAndTrack(output=output, VideoInput=args['VideoInput'],
                        cfg=cfg, data=data, weights=weights, prototxt=prototxt, model=model,
                        conf_thres=args['conf_thres'], nms_thres=args['nms_thres'], doubleCheckThres=args['doubleCheckThres'],
                        trackingConfidenceThres=args['trackingConfidenceThres'],
                        save_output=args['save_output'], show_output=args['show_output'], sliding_Detection=sliding_Detection,
                        State_Flag=args['Initial_State'],
                        RealTimeSimu=args['RealTimeSimu'], VerificationInterval=args['VerificationInterval'],
                        saveSmoothVideo=saveSmoothVideo, ImportType=args['ImportType'])

### RUN
detectorAndTracker.Run()





