# Date: Wednesday 07 June 2017 11:28:11 AM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: tracker manager

import time

import cv2
from goturn.helper.BoundingBox import BoundingBox

opencv_version = cv2.__version__.split('.')[0]

class tracker_manager:

    """Docstring for tracker_manager. """

    def __init__(self, regressor, tracker, logger):
        """This is

        :videos: list of video frames and annotations
        :regressor: regressor object
        :tracker: tracker object
        :logger: logger object
        :returns: list of video sub directories
        """
        self.regressor = regressor
        self.tracker = tracker
        self.logger = logger

    def trackAll(self, inputVideo,save_tracking=False,show_tracking=True):
        """Track the objects in the video
        """
        objRegressor = self.regressor
        objTracker = self.tracker

        vid = cv2.VideoCapture(inputVideo)

        ok, frame_0 =vid.read()
        if not ok:
            print('Couldnt read first frame')
            exit()

        if save_tracking:
            movie_number =0
            tracker_type ='GoturnGPU'
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            FPS = vid.get(cv2.CAP_PROP_FPS)
            output_video_name = 'output_videos/movie_' + str(movie_number) + time.strftime(
                "_%d%m%Y_%H%M%S") + '_' + tracker_type + '.avi'
            outVideo = cv2.VideoWriter(output_video_name, fourcc, int(FPS), (width, height))

        box_0 = cv2.selectROI(frame_0)
        cv2.destroyAllWindows()
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        bbox_0 = BoundingBox(x1=box_0[0], y1=box_0[1], x2=box_0[0] +box_0[2], y2=box_0[1] +box_0[3])
        objTracker.init(frame_0, bbox_0, objRegressor)
        
        timerI = cv2.getTickCount()
        for i in range(1, num_frames-1):
            # vid.set(1,i)
            print('Tracking on frame:'+str(i)+'/'+str(num_frames))
            ok,frame = vid.read()
            if not ok:
                break
            ### Start timer
            timer = cv2.getTickCount()

            ### Update Tracker
            bbox = objTracker.track(frame, objRegressor)

            ### Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            ### Draw bounding box
            ImageDraw = frame.copy()
            ImageDraw = cv2.rectangle(ImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)


            # Display FPS on frame
            cv2.putText(ImageDraw, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            if show_tracking:
                cv2.imshow('Results', ImageDraw)
                cv2.waitKey(1)
            if save_tracking:
                outVideo.write(ImageDraw)
        timerF = cv2.getTickCount()
        Time_proc = (timerF-timerI)/cv2.getTickFrequency()
        print('Total processing time ='+ str(int(Time_proc))+'sec')
        print('Average FPS ='+ str(i / Time_proc))
        vid.release()
        if save_tracking:
                outVideo.release()

    def trackAllGoturnKcf(self, inputVideo, save_tracking=False, show_tracking=True, Frame_int_Goturn=0):
        """Track the objects in the video
        Combining Opencv's KCF tracking with the goturn tracking.
        """
        objRegressor = self.regressor
        objTracker = self.tracker

        vid = cv2.VideoCapture(inputVideo)
        ok, frame_0 = vid.read()
        if not ok:
            print('Couldnt read first frame. Exit.')
            exit()

        if save_tracking:
            movie_number = 0
            tracker_type = 'GoturnGPU'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            FPS = vid.get(cv2.CAP_PROP_FPS)
            output_video_name = 'output_videos/movie_' + str(movie_number) + time.strftime(
                "_%d%m%Y_%H%M%S") + '_' + tracker_type + '.mp4'
            outVideo = cv2.VideoWriter(output_video_name, fourcc, int(FPS), (width, height))

        box_0 = cv2.selectROI(frame_0)
        cv2.destroyAllWindows()
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        bbox_0 = BoundingBox(x1=box_0[0], y1=box_0[1], x2=box_0[0] + box_0[2], y2=box_0[1] + box_0[3])
        # Init Goturn
        objTracker.init(frame_0, bbox_0, objRegressor)

        # Init KCF
        multiTracker = cv2.MultiTracker_create()
        multiTracker.add(cv2.TrackerKCF_create(), frame_0,
                         (bbox_0.x1, bbox_0.y1, (bbox_0.x2 -bbox_0.x1), (bbox_0.y2 -bbox_0.y1) ))
        num_KCF = 0
        bbox = bbox_0
        timerI = cv2.getTickCount()
        for i in range(2, num_frames):
            print('Tracking on frame:' + str(i) + '/' + str(num_frames))
            ok, frame = vid.read()
            if not ok:
                break
            ### Start timer
            timer = cv2.getTickCount()


            ### Update Tracker
            if num_KCF < Frame_int_Goturn:
                ok_KCF, box = multiTracker.update(frame)
                if ok_KCF:
                    bbox = BoundingBox(x1=box[0][0], y1=box[0][1], x2=box[0][0] + box[0][2], y2=box[0][1] + box[0][3])
                    num_KCF += 1
                    color = (50, 255, 50)
                if num_KCF == Frame_int_Goturn or not ok_KCF:
                    objTracker.update__prev_Jo(frame, bbox)
            if not num_KCF < Frame_int_Goturn or not ok_KCF:
                num_KCF = 0
                bbox = objTracker.track(frame, objRegressor)

                multiTracker = cv2.MultiTracker_create()
                multiTracker.add(cv2.TrackerKCF_create(), frame,
                                 (int(bbox.x1), int(bbox.y1), int(bbox.x2 - bbox.x1), int(bbox.y2 - bbox.y1) ))
                color = (255, 0, 0)


            ### Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            ### Draw bounding box
            ImageDraw = frame.copy()
            ImageDraw = cv2.rectangle(ImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
                                      color, 2)

            # Display FPS on frame
            cv2.putText(ImageDraw, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            if show_tracking:
                cv2.imshow('Results', ImageDraw)
                cv2.waitKey(1)
            if save_tracking:
                outVideo.write(ImageDraw)
        timerF = cv2.getTickCount()
        Time_proc = (timerF - timerI) / cv2.getTickFrequency()
        print('Total processing time =' + str(int(Time_proc)) + 'sec')
        print('Average FPS =' + str(i / Time_proc))
        cv2.imshow('Results', ImageDraw)
        cv2.waitKey(5000)
        vid.release()
        if save_tracking:
            outVideo.release()

    def trackAllJetsonCam(self, inputVideo, save_tracking=False, show_tracking=True, Frame_int_Goturn=0, TimeToRun=30,camera =False):
        """Track the objects in the video
        TimeToRun IN SECONDS !
        """
        objRegressor = self.regressor
        objTracker = self.tracker

        if camera:
            vid = cv2.VideoCapture(
                "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, "
                "framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink")
            num_frames = TimeToRun *20
        else:
            vid = cv2.VideoCapture(inputVideo)
            num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame_0 = vid.read()
        if not ok:
            print('Couldnt read first frame. Exit.')
            exit()
        FPS = vid.get(cv2.CAP_PROP_FPS)

        if save_tracking:
            movie_number = 903
            tracker_type = 'GoturnGPU'
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_name = 'output_videos/movie_' + str(movie_number) + time.strftime(
                "_%d%m%Y_%H%M%S") + '_' + tracker_type + '.avi'
            outVideo = cv2.VideoWriter(output_video_name, fourcc, int(60), (width, height))

        box_0 = cv2.selectROI(frame_0)
        cv2.destroyAllWindows()

        bbox_0 = BoundingBox(x1=box_0[0], y1=box_0[1], x2=box_0[0] + box_0[2], y2=box_0[1] + box_0[3])
        # Init Goturn
        objTracker.init(frame_0, bbox_0, objRegressor)

        # Init KCF
        multiTracker = cv2.MultiTracker_create()
        multiTracker.add(cv2.TrackerKCF_create(), frame_0,
                         (bbox_0.x1, bbox_0.y1, (bbox_0.x2 -bbox_0.x1), (bbox_0.y2 -bbox_0.y1) ))
        num_KCF = 0
        bbox = bbox_0
        timerI = cv2.getTickCount()
        for i in range(2, num_frames):
            print('Tracking on frame:' + str(i) + '/' + str(num_frames))
            ok, frame = vid.read()
            if not ok:
                break
            ### Start timer
            timer = cv2.getTickCount()


            ### Update Tracker
            if num_KCF < Frame_int_Goturn:
                ok_KCF, box = multiTracker.update(frame)
                if ok_KCF:
                    bbox = BoundingBox(x1=box[0][0], y1=box[0][1], x2=box[0][0] + box[0][2], y2=box[0][1] + box[0][3])
                    num_KCF += 1
                    color = (50, 255, 50)
                if num_KCF == Frame_int_Goturn or not ok_KCF:
                    objTracker.update__prev_Jo(frame, bbox)
            if not num_KCF < Frame_int_Goturn or not ok_KCF:
                num_KCF = 0
                bbox = objTracker.track(frame, objRegressor)

                multiTracker = cv2.MultiTracker_create()
                multiTracker.add(cv2.TrackerKCF_create(), frame,
                                 (int(bbox.x1), int(bbox.y1), int(bbox.x2 - bbox.x1), int(bbox.y2 - bbox.y1) ))
                color = (255, 0, 0)


            ### Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            ### Draw bounding box
            ImageDraw = frame.copy()
            ImageDraw = cv2.rectangle(ImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
                                      color, 2)

            # Display FPS on frame
            cv2.putText(ImageDraw, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            if show_tracking:
                cv2.imshow('Results', ImageDraw)
                cv2.waitKey(1)
            if save_tracking:
                outVideo.write(ImageDraw)
        timerF = cv2.getTickCount()
        Time_proc = (timerF - timerI) / cv2.getTickFrequency()
        print('Total processing time =' + str(int(Time_proc)) + 'sec')
        print('Average FPS =' + str(i / Time_proc))
        cv2.imshow('Results', ImageDraw)
        cv2.waitKey(5000)
        vid.release()
        if save_tracking:
            outVideo.release()

    def trackAllRealTimeSimu(self,input_folder, inputVideo, save_tracking=False, show_tracking=True, Frame_int_Goturn=0):
        """Track the objects in the video
        Simulates real time by taking the frame according to computation time of each itteration.
        """
        objRegressor = self.regressor
        objTracker = self.tracker

        vid = cv2.VideoCapture(input_folder+inputVideo)
        FPS = vid.get(cv2.CAP_PROP_FPS)
        ok, frame_0 = vid.read()
        if not ok:
            print('Couldnt read first frame. Exit.')
            exit()

        if save_tracking:

            movie_number = 0
            tracker_type = 'GoturnGPU'
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_name = 'output_videos/movie_'+inputVideo + str(movie_number) + time.strftime(
                "_%d%m%Y_%H%M%S") + '_' + tracker_type + '.avi'
            outVideo = cv2.VideoWriter(output_video_name, fourcc, int(FPS), (width, height))

        box_0 = cv2.selectROI(frame_0)
        cv2.destroyAllWindows()
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        bbox = BoundingBox(x1=box_0[0], y1=box_0[1], x2=box_0[0] + box_0[2], y2=box_0[1] + box_0[3])
        # Init Goturn
        objTracker.init(frame_0, bbox, objRegressor)

        # Init KCF
        multiTracker = cv2.MultiTracker_create()
        multiTracker.add(cv2.TrackerKCF_create(), frame_0,
                         (bbox.x1, bbox.y1, (bbox.x2 -bbox.x1), (bbox.y2 -bbox.y1) ))
        num_KCF = 0
        TimeSimulation = 0
        i=0
        color = (50, 255, 50)
        while True:
            i+=1
            # Simulate real time by taking the next frame
            TimeSimuFrame = TimeSimulation * FPS
            NumframeSimu = int(TimeSimuFrame)
            print('Tracking on frame:' + str(NumframeSimu) + '/' + str(num_frames))
            vid.set(1,NumframeSimu)
            ok, frame = vid.read()
            if not ok:
                break

            ### Start timer
            timeA = cv2.getTickCount()

            ### Update Tracker
            if num_KCF < Frame_int_Goturn:
                ok_KCF, box = multiTracker.update(frame)
                if ok_KCF:
                    bbox = BoundingBox(x1=box[0][0], y1=box[0][1], x2=box[0][0] + box[0][2], y2=box[0][1] + box[0][3])
                    num_KCF += 1
                if num_KCF == Frame_int_Goturn or not ok_KCF:
                    objTracker.update__prev_Jo(frame, bbox)
                    color = (255, 0, 0)
            if not num_KCF < Frame_int_Goturn or not ok_KCF:
                num_KCF = 0
                bbox = objTracker.track(frame, objRegressor)

                multiTracker = cv2.MultiTracker_create()
                multiTracker.add(cv2.TrackerKCF_create(), frame,
                                 (int(bbox.x1), int(bbox.y1), int(bbox.x2 - bbox.x1), int(bbox.y2 - bbox.y1) ))

            bbox = objTracker.track(frame, objRegressor)
            ### Calculate Frames per second (FPS)
            timeB = cv2.getTickCount()
            fps = cv2.getTickFrequency() /(timeB - timeA)
            TimeSimulation += 1/fps

            ### Draw bounding box
            ImageDraw = frame.copy()
            ImageDraw = cv2.rectangle(ImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
                                      color, 2)
            color = (50, 255, 50)

            # Display FPS on frame
            cv2.putText(ImageDraw, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            if show_tracking:
                cv2.imshow('Results', ImageDraw)
                cv2.waitKey(1)
            if save_tracking:
                outVideo.write(ImageDraw)
        timerF = cv2.getTickCount()
        print('Total processing time =' + str(TimeSimulation))
        print('Average FPS =' + str(i/TimeSimulation))
        if save_tracking:
            outVideo.release()
