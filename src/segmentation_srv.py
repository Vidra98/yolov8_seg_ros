from typing import Any
from yolov8_ros.srv import segmentationSrv, segmentationSrvResponse
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from segmentation import YOLOSegmentation
import yolov8_ros.srv
import numpy as np

from dynamic_reconfigure.server import Server
import yolov8_ros.cfg
from yolov8_ros.cfg import segmentationConfig
import cv2

class segGenerator():
    def __init__(self, param, node_name="/segmentation"):
        """Initialize the segmentation service, the service is called segmentation and receive a bgr image and return a black and white image with the segmentation idx
        """
        self.srv = rospy.Service('segmentation', segmentationSrv, self.handle_segmentation)
        self.bridge = CvBridge()
        self.model_size = param["model_size"]
        self.seg = YOLOSegmentation(model_size=self.model_size)

        # parameters
        self.conf = param["confidence_threshold"] # confidence threshold
        self.iou = param["iou_threshold"]  # NMS IoU threshold
        self.device = param["device"] # device to run the model on
        self.classes = [param["classes"]] #[41: 'cup', 46: 'banana', 47: 'apple', 49: 'orange', 50: 'brocoli', 51: 'carrot'].
        self.segmentation_type = param["segmentation_type"]
        self.draw = param["publish_segmap"]

        self.node = node_name
        # Window size to filter the segmentation
        self.width_range = [int(param["segWindowMinX"]), int(param["segWindowMaxX"])]
        self.height_range = [int(param["segWindowMinY"]), int(param["segWindowMaxY"])]

        #create the image publisher
        self.pub = rospy.Publisher('segmentation_image', Image, queue_size=2)

    def handle_segmentation(self, req):
        try:
            bgr = self.bridge.imgmsg_to_cv2(req.image, "passthrough")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            print(e)

        try:
            param = rospy.get_param(self.node)
            self.conf = param["confidence_threshold"]
            self.iou = param["iou_threshold"]
            self.classes = [param["classes"]]
            # classes = classes.split(" ")
            # self.classes = [int(i) for i in classes]
            self.device = param["device"]
            self.segmentation_type = param["segmentation_type"]
            self.width_range = [int(param["segWindowMinX"]), int(param["segWindowMaxX"])]
            self.height_range = [int(param["segWindowMinY"]), int(param["segWindowMaxY"])]
            self.draw = param["publish_segmap"]
            if param["model_size"] != self.model_size:
                self.model_size = param["model_size"]
                del self.seg
                self.seg = YOLOSegmentation(model_size=self.model_size)
                print("Model size changed to {}".format(self.model_size))

        except:
            rospy.logwarn("No parameters found, using default values")
        cv2.imwrite("/home/vdrame/test.png", rgb)
        # Call the segmentation model
        self.seg.predict(rgb, conf=self.conf, iou=self.iou, device=self.device, classes=self.classes)
        segmap = self.seg.get_segmap(segmentation_type=self.segmentation_type, classes=self.classes)

        if len(segmap.shape) == 3:
            segmap = segmap[0]

        if segmap is None:
            segmap = np.zeros(rgb.shape[:2]).astype(np.float64)
        
        segmap_size = segmap.shape

        # reshape img if not segmap size
        if segmap_size[0] != bgr.shape[0] or segmap_size[1] != bgr.shape[1]:
            segmap = cv2.resize(segmap, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.width_range and self.height_range:
            if min(self.width_range) > 0 and min(self.width_range) < segmap_size[1]:
                segmap[:, :min(self.width_range)] = 0
            if max(self.width_range) > 0 and max(self.width_range) < segmap_size[1]:
                segmap[:, max(self.width_range)+1:] = 0

            if min(self.height_range) > 0 and min(self.height_range) < segmap_size[0]:
                segmap[:min(self.height_range), :] = 0
            if max(self.height_range) > 0 and max(self.height_range) < segmap_size[0]:
                segmap[max(self.height_range)+1:, :] = 0

        #segmap in color
        if self.draw:
            colors = cv2.applyColorMap((segmap/np.max(segmap)*255).astype(np.uint8), cv2.COLORMAP_JET)
            segmap_clr = 0.7*bgr + 0.3*colors
            # Draw range limits
            cv2.rectangle(segmap_clr, (min(self.width_range), min(self.height_range)), (max(self.width_range), max(self.height_range)), (0, 0, 0), 2)
            for obj_idx in np.unique(segmap):
                if obj_idx == 0:
                    continue
                text_pos = np.mean(np.where(segmap == obj_idx), axis=1).astype(np.int32)
                # Write object index
                cv2.putText(segmap_clr, str(obj_idx), tuple((text_pos[1], text_pos[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                # Draw object center
                cv2.circle(segmap_clr, tuple((text_pos[1], text_pos[0])), 5, (255, 0, 0), -1)
            cv2.imwrite("/home/vdrame/test_seg.png", segmap_clr)
            segmap_clr = segmap_clr.astype(np.uint8)
            segmap_clr_msg = self.bridge.cv2_to_imgmsg(segmap_clr, "rgb8")
            self.pub.publish(segmap_clr_msg)

        #Convert to ros msg
        segmap_msg = self.bridge.cv2_to_imgmsg(segmap, "64FC1")
        return segmentationSrvResponse(segmap_msg)

def config_callback(config, level):
        # Update the parameters with the new values
        node = "/segmentation"
        print("segmentation reconfigure request")
        print("Reconfigure Request: ", config)
        rospy.set_param(node + '/confidence_threshold', config['confidence_threshold'])
        rospy.set_param(node + '/iou_threshold', config['iou_threshold'])
        rospy.set_param(node + '/classes', config['classes'])
        rospy.set_param(node + '/device', config['device'])
        rospy.set_param(node + '/segmentation_type', config['segmentation_type'])
        rospy.set_param(node + '/segWindowMinX', config['segWindowMinX'])
        rospy.set_param(node + '/segWindowMaxX', config['segWindowMaxX'])
        rospy.set_param(node + '/segWindowMinY', config['segWindowMinY'])
        rospy.set_param(node + '/segWindowMaxY', config['segWindowMaxY'])
        return config

if __name__ == '__main__':
    rospy.init_node('segmentation_server')
    node = "/segmentation"
    parameter = rospy.get_param(node)
    segGenerator(parameter)
    srv = Server(segmentationConfig, config_callback)
    rospy.spin()
