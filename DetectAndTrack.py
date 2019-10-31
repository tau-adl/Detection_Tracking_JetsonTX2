from goturn.logger.logger import setup_logger
from goturn.network.regressor import regressor
from goturn.tracker.tracker import tracker
from goturn.tracker.tracker_manager import tracker_manager
from yolo.detect import *
from DetectAndTrackUtils import *
from videoUtils import *

class DetectAndTrack :
    def __init__(self,
                 VideoInput = None,
                 output="ouptput",
                 cfg = 'yolo/model/yolo-drone.cfg',
                 data = 'yolo/model/yolo-drone.data',
                 weights = 'yolo/model/yolo-drone_100000_49.weights',
                 prototxt = 'goturn/nets/tracker.prototxt' ,
                 model = 'goturn/nets/tracker.caffemodel',
                 conf_thres = 0.8,
                 nms_thres = 0.15,
                 doubleCheckThres= 0.95,
                 trackingConfidenceThres=0.5,
                 save_output = False,
                 show_output = True,
                 sliding_Detection = False,
                 RealTimeSimu = False,
                 saveSmoothVideo = False,
                 ImportType =None,
                 VerificationInterval = 5,
                 State_Flag = "Detection"
                 ):
        """
        Initialization of the tracking and detection class.
        PARAMETERS:
        :param VideoInput:  Path to input video
        :param output:      Path to output folder
        :param cfg:         YOLO detection configuration file path (eg: yolo-drone.cfg)
        :param data:        YOLO detection data file path (eg: yolo-drone.data)
        :param weights:     YOLO detection weights file path (eg: yolo-drone.weights)
        :param prototxt:    GOTURN tracking prototxt file path (eg: solver.prototxt)
        :param model:       GOTURN tracking model path (eg: tracker.caffemodel)
        :param conf_thres:              YOLO detection confidence threshold for detection
        :param nms_thres:               YOLO detection Non-Maximal Suppression threshold => keep only best bounding box
        :param doubleCheckThres:        Confidence threshold above which we do not double check our detection
        :param trackingConfidenceThres: Confidence threshold above which we consider the tracker on desired object
        :param save_output:          Flag => True if you want to save the output
        :param show_output:          Flag => True if you want to save the output
        :param sliding_Detection:    Flag => True if you want to detect using a smaller search window, sliding on frame
        :param RealTimeSimu:         Flag => True if you want to simulate real time performances of algorithm
        :param saveSmoothVideo:      Flag => True to complete missing frames in output file to get a smooth video
        :param ImportType:           Type of import of the video ("JetsonCam" or "Queue" or "preLoadVideo")
        :param VerificationInterval: Time between two verification of accuracy of tracker [seconds]
        :param State_Flag:           Initial state of algorithm (eg: "Tracking" or "Detection")
        """
        ### Initialize Flags
        self.State_Flag = State_Flag
        self.save_output = save_output
        self.show_output = show_output
        self.sliding_Detection = sliding_Detection
        self.RealTimeSimu = RealTimeSimu
        self.saveSmoothVideo = saveSmoothVideo

        ### Initialize constants
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.doubleCheckThres = doubleCheckThres
        self.trackingConfidenceThres = trackingConfidenceThres

        self.VerificationInterval = VerificationInterval
        self.frameCount = 0

        ### Initialize simulation of real time
        self.TimeSimulation = 0
        self.time_trackingWV = 0    # Time passed since last verification
        self.trackingConfidence = 0
        self.meanfps = 30

        # Import input video
        if ImportType =="Pre-load":
            self.video = VideoReader(VideoInput,500)
        if ImportType =="Queue":
            self.video = VideoReaderQueue(VideoInput)
        elif ImportType =="JetsonCam":
            ### Jetson Camera
            self.video = cv2.VideoCapture(
                "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, "
                "framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink")
            ### Laptop webcam
            # self.video = cv2.VideoCapture(0)
            self.RealTimeSimu = False   # If on camera the real time doesn't need to be simulated
        else:
            self.video = cv2.VideoCapture(VideoInput)
        self.FPS = int(self.video.get(cv2.CAP_PROP_FPS))
        if self.FPS == 0:
            print('Error while importing video. Check video path.')
            exit()

        # Initialize detector
        self.detector = Frame_detecter(cfg, data, weights, output, img_size=416)
        self.detectionList = DetecterList()

        # Initialize tracker
        logger = setup_logger(logfile=None)
        self.objRegressor = regressor(prototxt, model, logger)
        self.objTracker = tracker(self.objRegressor)
        self.trackingList = TrackerList()

        self.numframes = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

        if self.State_Flag == "Tracking":
            ### If we start with tracking, GUI allowing to chose the object to track
            ok, frame_0 = self.video.read()
            self.frameCount +=1
            if not ok:
                if self.save_output:
                    self.outVideo.release()
                self.video.release()
                exit()
            box_0 = cv2.selectROI(frame_0)
            cv2.destroyAllWindows()
            bbox = BoundingBox(x1=box_0[0], y1=box_0[1], x2=box_0[0] + box_0[2], y2=box_0[1] + box_0[3])
            ### Init Goturn tracking
            self.objTracker.init(frame_0, bbox, self.objRegressor)
        if sliding_Detection:
            ### Computing frame sliding steps (made so that all cropped frames are 416x416)
            self.winStepHeight = int((self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) - 416) /
                                     (int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 416)))
            self.winStepWidth = int((self.video.get(cv2.CAP_PROP_FRAME_WIDTH) - 416) /
                                    (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH) / 416)))

        if self.save_output:
            ### Init of the outpput file
            inputName = VideoInput.split('/')[-1:][0].split('.')[:-1][0]
            output_video_name = 'output/' + inputName + '_DetectAndTrack' + time.strftime(
                "_%-Y-%m-%d_%H-%M") + '.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.saveSmoothVideo:
                self.outVideo = VideoSaver(output_video_name, fourcc, self.FPS, width, height)
                self.outVideo.AddIntroFrame(inputName, height, width, self.conf_thres, self.nms_thres,
                                            self.doubleCheckThres, self.trackingConfidenceThres,
                                            self.VerificationInterval, self.RealTimeSimu,
                                            weights)
            else:
                self.outVideo = cv2.VideoWriter(output_video_name, fourcc, self.FPS, (width, height))

    def track(self):
        """
        Tracking of the drone using caffe implementation of GoTurn Algorithm.
        Every time interval ("VerificationInterval"), YOLO detection is used to enhance tracking accuracy as well as
        to verify if the tracker hasn't lost the object.
        """
        if self.RealTimeSimu:
            ### Simulate real time by taking the next frame according to computation time past for last itteration.
            self.frameCount = round(self.TimeSimulation * self.FPS + 0.5)
            print('Tracking on frame:' + str(self.frameCount) + '/' + str(self.numframes))
            self.video.set(1, self.frameCount)
        ok, frame = self.video.read()
        self.frameCount += 1
        if not ok:
            if self.save_output:
                self.outVideo.release()
            self.video.release()
            exit()

        ### Start timer
        timeA = time.time()

        ### Update Tracker
        bbox = self.objTracker.track(frame, self.objRegressor)

        ### If you want to keep a list of all prev pos of tracker => uncomment next line
        # self.trackingList.add(self.frameCount-1, bbox, self.trackingConfidence)

        ### Calculate tracking speed Frames per second (FPS)
        timeB = time.time()
        self.meanfps = (self.meanfps*5 +1 / (timeB - timeA))/6
        self.TimeSimulation += timeB - timeA
        self.time_trackingWV += timeB - timeA
        print('Frame' + str(self.frameCount)+'   Tracking time: '+str(round(timeB - timeA,3)*1000)+'ms')

        if self.show_output or self.save_output:
            ImageDraw = plotTracking(frame, bbox, self.trackingConfidence, self.meanfps)
            if self.show_output:
                cv2.imshow('Results', ImageDraw)
                cv2.waitKey(1)
            if self.save_output:
                if self.saveSmoothVideo:
                    self.outVideo.write(ImageDraw, self.frameCount)
                else:
                    self.outVideo.write(ImageDraw)

        ### Check if tracker indeed on a drone
        if self.time_trackingWV > self.VerificationInterval:
            ok = False
            i = 0
            while not ok and i < 3:
                ### This loop allows to look for the drone in the next 3 frames.
                i +=1
                ok = self.VerifyTracker(bbox)
            if not ok:
                self.State_Flag = "Detection"
                print('Tracker Lost => State = Detection')

    def VerifyTracker(self, bbox_prev):
        ### Reinitialize the time since last verification to zero
        self.time_trackingWV = 0
        ### Read frame
        if self.RealTimeSimu:
            ### Simulate real time by taking the next frame according to computation time past for last itteration.
            self.frameCount = round(self.TimeSimulation* self.FPS + 0.5)
            print('Checking Detection on frame:' + str(self.frameCount) + '/' + str(self.numframes))
            self.video.set(1, self.frameCount)
        ok, frame = self.video.read()
        self.frameCount += 1
        if not ok:
            if self.save_output:
                self.outVideo.release()
            self.video.release()
            exit()

        ### Start timmer
        timeA = time.time()
        ### Detection on tracker's neighbourhood
        ### This neighbourhood has shown best results for our case but it could be different.
        x_min = int(max(bbox_prev.get_center_x()-min(bbox_prev.get_width()*4,208), 0))
        x_max = int(min(bbox_prev.get_center_x()+min(bbox_prev.get_width()*4,208), frame.shape[1]))
        y_min = int(max(bbox_prev.get_center_y()-min(bbox_prev.get_height()*4,208), 0))
        y_max = int(min(bbox_prev.get_center_y()+min(bbox_prev.get_height()*4,208), frame.shape[0]))
        croped_frame = frame[y_min:y_max, x_min:x_max]

        DetectFlag, BBox, self.trackingConfidence = self.detector.detectFrame(croped_frame,
                                                                              conf_thres=self.trackingConfidenceThres,
                                                                              nms_thres=self.nms_thres)
        ### Stop timmer
        timeB = time.time()
        self.TimeSimulation += timeB - timeA
        self.meanfps = (self.meanfps*5 +1/(timeB-timeA))/6
        print('Frame'+str(self.frameCount)+'  Verification of tracker:   '+str(round(timeB - timeA,3)*1000)+'ms')

        if DetectFlag:
            ### Adapt detection to the complete frame coordinates
            detecBBox = BBox[0]
            detecBBox.x1 += x_min
            detecBBox.x2 += x_min
            detecBBox.y1 += y_min
            detecBBox.y2 += y_min
            ### Update the tracker
            self.objTracker.update(frame, detecBBox)

        if self.show_output or self.save_output:
            if DetectFlag:
                ### Show tracker
                frame = plotTracking(frame, detecBBox, self.trackingConfidence, self.meanfps)

            if self.show_output:
                cv2.imshow('Results', frame)
                cv2.waitKey(1)
            if self.save_output:
                if self.saveSmoothVideo:
                    self.outVideo.write(frame, self.frameCount)
                else:
                    self.outVideo.write(frame)
        return DetectFlag

    def detect(self):
        """
        Detection based on YOLO detection algorithm.
        Step 1: Looks for a drone in the entire frame usually [1080,1920]
        Step 2: If didn't detect a drone => Scanning with a sliding window of 416*416
        """
        ### Read frame
        if self.RealTimeSimu:
            ### Simulate real time by taking the next frame according to computation time past for last itteration.
            self.frameCount = round(self.TimeSimulation * self.FPS + 0.5)
            print('Detecting on frame:' + str(self.frameCount) + '/' + str(self.numframes))
            self.video.set(1, self.frameCount)
        ok, frame = self.video.read()
        self.frameCount += 1
        if not ok:
            if self.save_output:
                self.outVideo.release()
            self.video.release()
            exit()

        ### Start timer
        timeA = time.time()

        ### Detecting on the complete frame
        DetectFlag, BBox, confidence = self.detector.divideAndDetectFrame(frame,
                                                                          conf_thres=self.conf_thres,
                                                                          nms_thres=self.nms_thres,
                                                                          slidingFlag=False,
                                                                          show_image=False)

        ### Stop timer
        timeB = time.time()
        self.TimeSimulation += timeB - timeA
        print("Detection time :" +str(round(timeB - timeA,2))+'sec')

        if DetectFlag:
            ### If drone detected => Initialize the tracking and switches the State to "Tracking"
            bbox =BBox[0]
            ### If you want to keep a list of all detection uncomment next line.
            # self.detectionList.add(self.frameCount-1,bbox, confidence)
            ### Initialize tracker
            self.trackingConfidence = confidence
            self.objTracker.init(frame, bbox, self.objRegressor)
            self.State_Flag ="Tracking"
            if confidence[0] < self.doubleCheckThres:
                ### If the confidence is smaller than "doubleCheckThres" => recheck on next frame if it is a drone indeed
                self.time_trackingWV = self.VerificationInterval
            else:
                self.time_trackingWV = 0

        if self.show_output or self.save_output:
            if DetectFlag:
                ImageDraw = plotDetection(frame, DetectFlag, bbox=bbox, confidence = confidence[0])
            else:
                ImageDraw = plotDetection(frame, DetectFlag)
            if self.show_output:
                cv2.imshow('Results', ImageDraw)
                cv2.waitKey(1)
            if self.save_output:
                if self.saveSmoothVideo:
                    self.outVideo.write(ImageDraw, self.frameCount)
                else:
                    self.outVideo.write(ImageDraw)

        ### If a drone wasn't detected on complete frame, scan the frame with sliding window.
        if not DetectFlag:
            DetectFlag, BBox, confidence = self.detectSlidingWinRealTime(frame, conf_thres=self.conf_thres,
                                                                              nms_thres=self.nms_thres,
                                                                              show_image=False)

    def detectSlidingWinRealTime(self,
                      img_size=416,     # Size of Image for Yolo_network
                      conf_thres=0.5,   # Confidence threshold
                      nms_thres=0.5,    # Non-Maximal Suppression Threshold
                      SlidingWinSize =416, #Sliding window size recommanded to use yolo network size win.
                      save_images=False,
                      show_image=False):
        """
        This function will scan the complete frame in order to look for a drone. The smaller croped image allows
        increased accuracy in detection.
        This function scans in real time, meaning each subframe is croped from the current frame.
        :param img_size:
        :param conf_thres:
        :param nms_thres:
        :param SlidingWinSize:
        :param save_images:
        :param show_image:
        :return:
        """
        ### Loop initialization
        if self.RealTimeSimu:
            ### Simulate real time by taking the next frame according to computation time past for last itteration.
            self.frameCount = round(self.TimeSimulation * self.FPS+0.5)
            print('Detecting on frame:' + str(self.frameCount) + '/' + str(self.numframes))
            self.video.set(1, self.frameCount)
        ok, frame = self.video.read()
        self.frameCount +=1
        if not ok:
            if self.save_output:
                self.outVideo.release()
            self.video.release()
            exit()

        DetectFlag = False
        cropWidth = 0

        ### Scanning
        while cropWidth + SlidingWinSize <= frame.shape[1] and not DetectFlag:
            cropHeight = 0
            while cropHeight + SlidingWinSize <= frame.shape[0] and not DetectFlag:
                ### Read current frame
                if self.RealTimeSimu:
                    ### Simulate real time by taking the next frame according to computation time past for last itteration.
                    self.frameCount = round(self.TimeSimulation * self.FPS +0.5)
                    print('Detecting on frame:' + str(self.frameCount) + '/' + str(self.numframes))
                    self.video.set(1, self.frameCount)
                ok, frame = self.video.read()
                self.frameCount += 1
                if not ok:
                    if self.save_output:
                        self.outVideo.release()
                    self.video.release()
                    exit()
                ### Start timer
                timeA = time.time()
                ### Cropping frame
                frame_cropped = frame[cropHeight : cropHeight+SlidingWinSize, cropWidth : cropWidth+SlidingWinSize]
                ### Detect on cropped frame
                DetectFlag, detecBBox, confidence = self.detector.detectFrame(frame_cropped,
                                                                     show_image=show_image,save_images=save_images,
                                                                     conf_thres = conf_thres, nms_thres= nms_thres)
                ### Stop timer
                timeB = time.time()
                self.TimeSimulation += timeB -timeA
                self.meanfps = (self.meanfps*5 + 1 / (timeB - timeA)) / 6
                print( 'subframe: ['+str(cropHeight)+','+str(cropWidth)+']'+"  Detection time for subframe:"+
                       str(round(timeB -timeA,3)*1000)+ 'ms')
                if DetectFlag:
                    ### Adapt Bounding box to the non-cropped frame coordinates
                    detecBBox[0].x1 += cropWidth
                    detecBBox[0].x2 += cropWidth
                    detecBBox[0].y1 += cropHeight
                    detecBBox[0].y2 += cropHeight

                if self.show_output or self.save_output:
                    slidingFrame = showSlidingWindow(frame, cropWidth, cropHeight, SlidingWinSize, self.meanfps)
                    if DetectFlag:
                        # Add bbox to the image with label
                        label = '%s %.2f' % ('Drone', confidence[0])
                        plot_one_box([detecBBox[0].x1, detecBBox[0].y1, detecBBox[0].x2, detecBBox[0].y2], slidingFrame, label=label)
                    if self.show_output:
                        cv2.imshow('Results', slidingFrame)
                        cv2.waitKey(1)
                    if self.save_output:
                        if self.saveSmoothVideo:
                            self.outVideo.write(slidingFrame, self.frameCount)
                        else:
                            self.outVideo.write(slidingFrame)

                cropHeight += self.winStepHeight  # 332
            cropWidth += self.winStepWidth  # 376

        if DetectFlag:
            ### If detection => initialize tracking and switch state to "Tracking"
            bbox = detecBBox[0]
            ### If you want to keep a list of all detection uncomment next line.
            # self.detectionList.add(self.frameCount-1,bbox, confidence)
            ### Initialize tracker
            self.objTracker.init(frame, bbox, self.objRegressor)
            self.State_Flag ="Tracking"
            if confidence[0] < self.doubleCheckThres:
                ### If the confidence is smaller than "doubleCheckThres" => recheck next frame if it is a drone indeed
                self.time_trackingWV = self.VerificationInterval
            else:
                self.time_trackingWV = 0

        return DetectFlag, detecBBox, confidence


    def Run(self):
        """
        Main Running Loop
        """
        while True:
            if self.State_Flag == "Detection":
                self.detect()
            elif self.State_Flag == "Tracking":
                self.track()
