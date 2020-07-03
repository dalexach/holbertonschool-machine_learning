#!/usr/bin/env python3
"""
Yolo class
"""

import tensorflow.keras as K
import numpy as np


class Yolo():
    """
    Class Yolo that uses the Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor of the class

        Arguments:
         - model_path is the path to where a Darknet Keras model is stored
         - classes_path is the path to where the list of class names used for
            the Darknet model, listed in order of index, can be found
         - class_t is a float representing the box score threshold for
            the initial filtering step
         - nms_t is a float representing the IOU threshold for
            non-max suppression
         - anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            * outputs is the number of outputs (predictions) made by
                the Darknet model
            * anchor_boxes is the number of anchor boxes used for
                each prediction
            * 2 => [anchor_box_width, anchor_box_height]

        Public instance attributes:
         - model: the Darknet Keras model
         - class_names: a list of the class names for the model
         - class_t: the box score threshold for the initial filtering step
         - nms_t: the IOU threshold for non-max suppression
         - anchors: the anchor boxes
        """

        model = K.models.load_model(model_path)
        class_names = []
        with open(classes_path) as f:
            class_names = f.readlines()
        for cn in class_names:
            class_names = cn.strip()

        self.model = model
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoidf(self, x):
        """
        Function that calculates sigmoid
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    # public method
    def process_outputs(self, outputs, image_size):
        """
        Public method to process the outputs

        Arguments:
         - outputs is a list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image:
            Each output will have the shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
             * grid_height & grid_width => the height and width of
                the grid used for the output
             * anchor_boxes => the number of anchor boxes used
             * 4 => (t_x, t_y, t_w, t_h)
             * 1 => box_confidence
             * classes => class probabilities for all classes
         - image_size is a numpy.ndarray containing the image’s original size
            [image_size[0], image_size[1]]

        Returns:
         A tuple of (boxes, box_confidences, box_class_probs):
         - boxes: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output:
            * 4 => (x1, y1, x2, y2)
            * (x1, y1, x2, y2) should represent the boundary box
                relative to original image
         - box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing the box confidences for each output, respectively
         - box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing the box’s class probabilities
            for each output, respectively
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            # Create the list with np.ndarray (grid_h, grid_w, anchor_boxes, 4)
            boxes.append(output[..., 0:4])

            # Calculate confidences for each output
            box_confidences.append(self.sigmoidf(output[..., 4:5]))

            # Calculate class probability for each output
            box_class_probs.append(self.sigmoidf(output[..., 5:]))

        for i, box in enumerate(boxes):
            grid_height = box.shape[0]
            grid_width = box.shape[1]
            anchor_box = box.shape[2]

            c = np.zeros((grid_height, grid_width, anchor_box), dtype=int)

            # Cy matrix
            idx_y = np.arange(grid_height)
            idx_y = idx_y.reshape(grid_height, 1, 1)
            Cy = c + idx_y

            # Cx matrix
            idx_x = np.arange(grid_width)
            idx_x = idx_x.reshape(1, grid_width, 1)
            Cx = c + idx_x

            # Center coordinates output and normalized
            tx_n = self.sigmoidf(box[..., 0])
            ty_n = self.sigmoidf(box[..., 1])

            # Calculate bx & by and normalize it
            bx = tx_n + Cx / grid_width
            by = ty_n + Cy / grid_height

            # Calculate tw & th
            tw = np.exp(box[..., 2])
            th = np.exp(box[..., 3])

            # Anchor box dimension
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # input size
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value

            # Calculate bw & bh and normalize
            bw = pw * tw / input_width
            bh = ph * th / input_height

            # Corner coordinates
            x1 = bx - bw / 2
            y1 = bh - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            # Adjust scale
            box[..., 0] = x1 * image_size[1]
            box[..., 1] = y1 * image_size[0]
            box[..., 2] = x2 * image_size[1]
            box[..., 3] = y2 * image_size[0]

        return (boxes, box_confidences, box_class_probs)


    # Public method
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Public method to filter the boxes

        Arguments:
         - boxes: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output, respectively
         - box_confidences: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, 1)
            containing the processed box confidences for each output, respectively
         - box_class_probs: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, classes)
            containing the processed box class probabilities for each output, respectively
        Returns:
         A tuple of (filtered_boxes, box_classes, box_scores):
         * filtered_boxes: a numpy.ndarray of shape (?, 4) containing
            all of the filtered bounding boxes:
         * box_classes: a numpy.ndarray of shape (?,) containing
            the class number that each box in filtered_boxes predicts, respectively
         * box_scores: a numpy.ndarray of shape (?) containing
            the box scores for each box in filtered_boxes, respectively
        """

        filtered_boxes = []
        box_classes = []
        box_scores = []
        scores = []

        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            scores.append(box_conf * box_class_prob)

        for score in scores:
            box_score = score.max(axis=-1)
            box_score = box_score.reshape(-1)
            box_scores.append(box_score)

            box_class = np.argmax(score, axis=-1)
            box_class = box_class.reshape(-1)
            box_classes.append(box_class)

        box_scores = np.concatenate(box_scores, axis=-1)
        box_classes = np.concatenate(box_classes, axis=-1)

        for box in boxes:
            filtered_boxes.append(box.reshape(-1, 4))

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        filtering_mask = np.where(box_scores >= self.class_t)

        filtered_boxes = filtered_boxes[filtering_mask]
        box_classes = box_classes[filtering_mask]
        box_scores = box_scores[filtering_mask]

        return (filtered_boxes, box_classes, box_scores)
