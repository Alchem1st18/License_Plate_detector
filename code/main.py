from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car,read_license_plate,write_csv

vehicle_tracker = Sort()

results = {}

#loading model
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('C:/Users/Srikar/PycharmProjects/car_plate_detection/models/plate-number-best.pt')

#loading video
captured_video = cv2.VideoCapture('C:/Users/Srikar/PycharmProjects/car_plate_detection/video/2103099-uhd_3840_2160_30fps.mp4')

frame_num = -1
vehicles = [2,3,5,7]
#reading frames
ret = True
while ret:
    frame_num+=1
    ret,frame = captured_video.read()
    if ret:
        results[frame_num] = {}
        #detecting objects
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = detection
            #print(x1,y1,x2,y2,score,class_id)
            #detecting vehicles specifically from all objects
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score])

        #tracking the vehicles detected
        track_ids = vehicle_tracker.update(np.asarray(detections_))

        #detecting license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = license_plate


            #assigning cars to license plates
            xcar1,ycar1,xcar2,ycar2,car_id = get_car(license_plate,track_ids)

            #cropping license plates
            license_plate_crop = frame[int(y1):int(y2),int(x1):int(x2),:]

            #process the license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY)
            _,license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray,64,255,cv2.THRESH_BINARY_INV)

            #read license plate number
            lp_text,lp_confidence = read_license_plate(license_plate_crop_thresh)

            if lp_text is not None:
                results[frame_num][car_id] = {'car':{'bbox':[xcar1,ycar1,xcar2,ycar2]},
                                              'license_plate':{'bbox':[x1,y1,x2,y2],'text':[lp_text],'bbox_score':score,'text_score':lp_confidence}}


#writing down the results
write_csv(results,'C:/Users/Srikar/PycharmProjects/car_plate_detection/results/test_result_values.csv')

