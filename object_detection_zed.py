import cv2
import numpy as np
import pyzed.sl as sl

#Create a Camera object
zed = sl.Camera()

#Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

#Open zed camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

#Set object detection parameters
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = True                                        #Objects will keep the same ID between frames
obj_param.enable_segmentation = True                                    #Outputs 2D masks over detected objects
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

if obj_param.enable_tracking:
    #Set positional tracking parameters
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    #Enable positional tracking
    zed.enable_positional_tracking(positional_tracking_parameters)

#Enable object detection with initialization parameters
zed_error = zed.enable_object_detection(obj_param)
if zed_error != sl.ERROR_CODE.SUCCESS:
    print("enable_object_detection", zed_error, "\nExit program.")
    zed.close()
    exit(-1)

#Create object container and runtime parameters
objects = sl.Objects()                                                  #Structure containing all the detected objects
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
obj_runtime_param.detection_confidence_threshold = 30

cv2.namedWindow("ZED 2i Camera", cv2.WINDOW_AUTOSIZE)

#Continuously grab and process frames
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        #Retrieve detected objects
        zed.retrieve_objects(objects, obj_runtime_param)

        #Create an image container and retrieve the left view image
        img = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)
        img_cv = img.get_data()

        #If new objects have been detected, process them
        if objects.is_new:
            obj_list = objects.object_list
            print(str(obj_list))
            for obj in obj_list:
                top_left = obj.bounding_box_2d[0]
                bottom_right = obj.bounding_box_2d[2]

                #Draw a rectangle around the detected object
                cv2.rectangle(
                    img_cv,
                    (int(top_left[0]), int(top_left[1])),
                    (int(bottom_right[0]), int(bottom_right[1])),
                    (0, 255, 0),
                    2
                )
                label = f"{obj.label}  ({int(obj.confidence * 100)}%)"

                #Add text
                cv2.putText(
                    img_cv, label,
                    (int(top_left[0]), int(top_left[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2
                )

        cv2.imshow("Object Detection using ZED 2i", img_cv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.disable_object_detection()
zed.close()
cv2.destroyAllWindows()