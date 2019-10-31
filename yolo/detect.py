import time
from sys import platform
import cv2
import torch

from yolo.models import *
from yolo.utils.datasets import *
from yolo.utils.utils import *
import yolo.utils.torch_utils as torch_utils
from videoUtils import *

from goturn.helper.BoundingBox import *


def detection_video(videoPath,
                             cfg='model/yolo-drone.cfg',
                             data='model/yolo-drone.data',
                             weights='model/yolo-drone_80000_49.weights',
                             output='output', save_output = False, show_output = False):
    # Initialize detector
    detector = Frame_detecter(cfg, data, weights, output, img_size=416)

    # Read Video
    # vid = cv2.VideoCapture(videoPath)
    vid = VideoReaderQueue(videoPath)
    num_Frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = vid.get(cv2.CAP_PROP_FPS)
        if not os.path.exists(output):
            os.makedirs(output)  # make new output folder
        output_video_name = 'output/Detect_' + time.strftime("_%d%m%Y_%H%M%S") +'.mp4'
        outVideo = cv2.VideoWriter(output_video_name, fourcc, int(FPS), (width, height))

    for i in range(0, num_Frames):
        print("detecting on frame:" + str(i) + '/' + str(num_Frames))
        ok, frame = vid.read()
        if not ok:
            break
        DetectFlag, BBox, confidence = detector.divideAndDetectFrame(frame, conf_thres = 0.5,nms_thres=0.15,
                                                                     slidingFlag =False, show_image=False)

        if DetectFlag:
            bbox =BBox[0]
            # Add bbox to the image with label
            label = '%s %.2f' % ('Drone', confidence[0])
            plot_one_box([bbox.x1, bbox.y1, bbox.x2, bbox.y2], frame, label=label)
        if save_output:
            outVideo.write(frame)
        if show_output:
            cv2.imshow('Output',frame)
            cv2.waitKey(1)
    if save_output:
        outVideo.release()
    vid.release()

class Frame_detecter:
    def __init__(self, cfg, data, weights, output, img_size = 416):
        # Initialize
        self.device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
        torch.backends.cudnn.benchmark = False  # set False for reproducible results
        # Create output dir if doesn't exist
        if not os.path.exists(output):
            os.makedirs(output)  # make new output folder
        # Initialize model
        self.model = Darknet(cfg, img_size)
        # Load weights
        _ = load_darknet_weights(self.model, weights)
        # Eval mode
        self.model.to(self.device).eval()
        # Get classes and colors
        self.classes = load_classes(parse_data_cfg(data)['names'])
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

    def detectFrame(self,
                    im0,              # frame
                    output='output',  # output folder
                    img_size=416,
                    conf_thres=0.5,   # Confidence threshold
                    nms_thres=0.5,    # Non-Maximal Suppression Threshold
                    save_images=False,
                    show_image=False):

        DetectFlag =False
        BBox = []
        confidence = []

        # Run inference
        time_0 = time.time()

        # Padded resize
        img, *_ = letterbox(im0, new_shape=img_size)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # converts from 0 - 255 to 0.0 - 1.0

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            DetectFlag = True
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            # print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('Detecting: '+'%g %ss' % (n, self.classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                BBox.append(BoundingBox(x1= int(det.data[0][0]), y1= int(det.data[0][1]), x2= int(det.data[0][2]), y2= int(det.data[0][3])))
                confidence.append( float(det.data[0][4]))
                if show_image or save_images:
                    # Add bbox to the image
                    label = '%s %.2f' % (self.classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=self.colors[0])
        if show_image:
            cv2.imshow('', im0)
            cv2.waitKey(500)
        if save_images:
            cv2.imwrite(output+'/frame_Detect_'+time.strftime("_%d%m%Y_%H%M%S")+'.jpg',im0)
        return DetectFlag, BBox, confidence


    def divideAndDetectFrame(self,
                      frame,
                      img_size=416,     # Size of Image for Yolo_network
                      conf_thres=0.5,   # Confidence threshold
                      nms_thres=0.5,    # Non-Maximal Suppression Threshold
                      slidingFlag=False,  # If you want to use a sliding window to enhance detection
                      SlidingWinSize =416, #Sliding window size recommanded to use yolo network size win.
                      save_images=False,
                      show_image=False):

        time_0 = time.time()
        if (frame.shape[0] > img_size or frame.shape[1] > img_size) and slidingFlag :
            # Computing frame sliding steps (made so that all cropped frames are 416x416)
            winStepHeight = int((frame.shape[0] - SlidingWinSize) / (int(frame.shape[0] / SlidingWinSize)))
            winStepWidth = int((frame.shape[1] - SlidingWinSize ) / (int(frame.shape[1] / SlidingWinSize)))

            # Loop initialize
            DetectFlag = False
            cropWidth = 0
            while cropWidth + SlidingWinSize <= frame.shape[1] and not DetectFlag:
                cropHeight = 0
                while cropHeight + SlidingWinSize <= frame.shape[0] and not DetectFlag:
                    # Cropping frame
                    frame_cropped = frame[cropHeight : cropHeight+SlidingWinSize, cropWidth : cropWidth+SlidingWinSize]
                    # Detect on cropped frame
                    DetectFlag, detecBBox, confidence = self.detectFrame(frame_cropped,
                                                                         show_image=show_image,save_images=save_images,
                                                                         conf_thres = conf_thres, nms_thres= nms_thres)
                    if DetectFlag :
                        #Adapt Bounding box to the non-cropped frame
                        detecBBox[0].x1 += cropWidth
                        detecBBox[0].x2 += cropWidth
                        detecBBox[0].y1 += cropHeight
                        detecBBox[0].y2 += cropHeight
                    cropHeight += winStepHeight  # 332
                cropWidth += winStepWidth  # 376

        else:
            DetectFlag, detecBBox, confidence = self.detectFrame(frame, show_image=show_image,save_images=save_images,
                                                                 conf_thres=conf_thres, nms_thres=nms_thres )
        return DetectFlag, detecBBox, confidence


