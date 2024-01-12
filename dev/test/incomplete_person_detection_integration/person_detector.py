import argparse
import os
import platform
import sys
from pathlib import Path

import torch

from ultralytics.utils.plotting import Annotator

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, strip_optimizer)
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class PersonDetector:
    """
    Class to detect persons in images and identify the coordinates
    of the bounding box around the detected person.
    """
    def __init__(self):
        self.hide_conf = True
        self.iou_thres = 0.45,  # NMS IOU threshold
        self.max_det = 1000,  # maximum detections per image
        self.view_img = False,  # show results
        self.classes = None,  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False,  # class-agnostic NMS
        self.augment = False,  # augmented inference
        self.line_thickness = 3,  # bounding box thickness (pixels)
        self.vid_stride = 1,  # video frame-rate stride


    @smart_inference_mode()
    def run(
        self,
        f_input,  # Frame read from video capture or other source
        weights=ROOT / 'yolov5m.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.7,  # confidence threshold
        hide_conf=False,  # hide confidences
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    ):
        """
        Run person detection using YOLOv5 model.
        """
        # Load model
        device = select_device(device)
        print("device = ")
        print(device)
        model = DetectMultiBackend(weights, device=device)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        dataset = LoadStreams(f_in=f_input, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        bs = len(dataset)

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = model(im, augment=self.augment, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, max_det=1000)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
                
                # p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                
                annotator = Annotator(im0, line_width=3, example=str(names))

                predictions = []  # Store predictions for each person
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    # Check if the detected object is a person (class index 0 for person)
                    if names[c] == 'person':
                        # Extract ROI coordinates and print to terminal
                        roi_coordinates = [int(coord) for coord in xyxy]
                        print(f"ROI Coordinates for Person {len(predictions) + 1}: {roi_coordinates}")  # print ROI coordinates for person detection
                        predictions.append((label, confidence_str, roi_coordinates))
                        
                return predictions, annotator.result()
                # Stream results
            #     im0 = annotator.result()
            #     if view_img:
            #         if platform.system() == 'Linux' and p not in windows:
            #             windows.append(p)
            #             cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #             cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            #         cv2.imshow(str(p), im0)
            #         key = cv2.waitKey(1)  # 1 millisecond
            #         if key in {ord('q'), ord('Q'), 27}:  # 'q' 키 또는 ESC를 누르면 종료
            #             break_key = True
            #             break
            # if break_key is True:
            #     break
                        
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            
