# YOLOv8 Segmentation Service

This repository contains a ROS service for performing image segmentation using YOLOv8. The service takes a BGR image as input and returns a black and white image with segmentation indices.

## Installation

1. Clone this repository to your ROS workspace:
   ```
   git clone https://github.com/Vidra98/yolov8_seg_ros.git
   ```

2. Install [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu) and Build the ROS package:
   ```
   cd <ros_workspace>
   catkin build yolov8_seg_ros
   ```

3. Install ROS the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Launch the segmentation service node:
   ```
   roslaunch yolov8_ros segmentation_rqt.launch
   ```

2. Subscribe to the segmentation service by calling the `segmentation` service with a BGR image message.

3. The service will perform segmentation on the input image and return a black and white image with segmentation indices.

4. Optionally, you can configure the segmentation parameters by using the dynamic reconfigure server. Run the following command to launch the dynamic reconfigure GUI:
   ```
   rosrun rqt_reconfigure rqt_reconfigure
   ```

5. In the GUI, select the `segmentation` node and adjust the parameters according to your requirements.

## Parameters

The following parameters can be configured:

- `confidence_threshold`: Confidence threshold for object detection.
- `iou_threshold`: IoU threshold for non-maximum suppression.
- `device`: Device to run the model on (e.g., "cpu", "cuda").
- `classes`: List of classes to perform segmentation on.
- `segmentation_type`: Segmentation type (e.g., "bbox", "mask").
- `segWindowMinX`, `segWindowMaxX`: Range limits for the segmentation window on the x-axis.
- `segWindowMinY`, `segWindowMaxY`: Range limits for the segmentation window on the y-axis.
- `publish_segmap`: Flag to publish the segmented image with colored overlays.

## Contributing

This repository consist simply of a wrapper for the implementation of Ultralytics work, contributions to this repository are welcome. Please open an issue or submit a pull request with your suggestions, bug reports, or feature enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

Feel free to modify the content as needed to suit your specific project.