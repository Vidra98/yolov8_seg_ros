from ultralytics import YOLO
import cv2
import numpy as np

class YOLOSegmentation:
    def __init__(self, model_size="l"):
        self._model = YOLO('segmentation/yolov8{}-seg.pt'.format(model_size))  # load a pretrained YOLOv8n segmentation model model are between [n, s, m, l, x]
        print("YOLOSegmentation init done")

    def predict(self, image, conf=0.2, iou = 0.7, device="cpu", classes=[41, 46, 47, 49, 50, 51]):
        """ Predict the segmentation of an image

        Args:
            image (_type_): RGB image to generate segmentation from
            conf (float, optional): Confidence threshold to consider select a prediction. Defaults to 0.2.
            iou (float, optional): intersection over union (IoU) threshold for non maximum suppression. Defaults to 0.7.
            device (str, optional): Device used for inference. Defaults to "cpu".
            classes (list, optional): Object to segment. Defaults to [41: 'cup' 46: 'banana', 47: 'apple', 49: 'orange', 50: 'broccoli', 51: 'carrot'].
        """
        self._input_shape = image.shape
        image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_LINEAR)
        self._preds = self._model.predict(image, conf = conf, iou = iou, device = device,
                                         classes=classes, show_conf=False, show_labels=False, 
                                         box=False, show=False, save=False, save_txt=False)  # predict on an image
        
    def get_pred(self):
        return self._preds

    def get_segmap(self, segmentation_type="class", classes=[47]):
        for pred in self._preds:

            if pred.masks is not None:
                # if not pred:
                #     return None
                (nb_obj, width, height) = pred.masks.data.shape
                segmap = np.zeros((width, height))
                # {None, instance, class, semantic}
                if segmentation_type == "semantic":
                    # Returns a 2D array containing up to 255 semantics
                    cls_ = pred.boxes.cls
                    for i, mask in enumerate(pred.masks.data.cpu().numpy()):
                        segmap = np.where(mask, cls_[i], segmap)
                    segmap = cv2.resize(segmap, dsize=(self._input_shape[1], self._input_shape[0]), interpolation=cv2.INTER_NEAREST)

                elif segmentation_type == "class":
                    # Returns the class of the object
                    if type(classes) == list or type(classes) == np.ndarray:
                        if  len(classes) > 1:
                            print("Please provide a single class to segment")
                            return None
                        else:
                            classes = classes[0]

                    cls_ = pred.boxes.cls
                    where = np.argwhere(pred.boxes.cls == classes)[0]
                    for i, mask in enumerate(pred.masks.data[where].cpu().numpy()):
                        segmap = np.where(mask, cls_[i], segmap)
                    segmap = cv2.resize(segmap, dsize=(self._input_shape[1], self._input_shape[0]), interpolation=cv2.INTER_NEAREST)

                elif segmentation_type == "instance":
                    # Returns a list of segmaps, one for each instance, in the order of the classes list, each segmap is a 2D array containing up to 255 semantics
                    if type(classes) != list and type(classes) != np.ndarray:
                        classes = [classes]

                    segmap = []
                    for selected_class in classes:
                        where = np.argwhere(pred.boxes.cls.cpu() == selected_class)[0]

                        if len(where) == 0:
                            print("No object of class {} found".format(selected_class))
                            continue
                        pred_np = pred.masks.data[where].cpu().numpy().copy()
                        pred_fuse = np.zeros(pred_np[0].shape)
                        for i in range(len(pred_np)):
                            pred_fuse = np.where(pred_np[i], i+1, pred_fuse)
                            if i+1 == 255:
                                print("Warning: More than 255 instances of class {} found".format(selected_class))
                                break
                        pred_fuse = cv2.resize(pred_fuse, dsize=(self._input_shape[1], self._input_shape[0]), interpolation=cv2.INTER_NEAREST)
                        segmap.append(pred_fuse)
                        segmap = np.array(segmap)
                else :
                    print("No segmentation type chosen, please choose between instance, class, semantic")
                    return np.zeros((self._input_shape[1], self._input_shape[0]))
            else:
                print("No object found")
                if segmentation_type == "instance":
                    return np.zeros((1, self._input_shape[0], self._input_shape[1]))
                return np.zeros((self._input_shape[0], self._input_shape[1]))
        return segmap

