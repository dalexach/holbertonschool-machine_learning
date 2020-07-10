#!/usr/bin/env python3
"""
Class FaceAlign
"""

import cv2
import dlib
import numpy as np


class FaceAlign:
    """
    FaceAlign class
    """

    def __init__(self, shape_predictor_path):
        """
        Constructor

        Arguments:
         - shape_predictor_path is the path to the dlib shape predictor model

        Public instance attributes:
         - detector - contains dlib‘s default face detector
         - shape_predictor - contains the dlib.shape_predictor
        """

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        Public instance method that detects a face in an image:

        Arguments:
         - image is a numpy.ndarray of rank 3 containing an image
            from which to detect a face

        Returns:
         A dlib.detectedangle containing the boundary box for the face in
            the image, or None on failure
        """

        h, w, c = image.shape
        detectedangle = dlib.detectedangle(left=0, top=0, right=w, bottom=h)
        area = 0
        try:
            detected = self.detector(image, 1)

            for d in detected:
                new_area = d.area()
                if new_area > area:
                    area = new_area
                    detectedangle = d

            return detectedangle

        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """
        Public instance method that finds facial landmarks

        Arguments:
         - image is a numpy.ndarray of an image from which to
            find facial landmarks
         - detection is a dlib.detectedangle containing the boundary box
            of the face in the image

        Returns:
         A numpy.ndarray of shape (p, 2)containing the landmark points,
         or None on failure
             - p is the number of landmark points
             - 2 is the x and y coordinates of the point
        """

        try:
            shape = self.shape_predictor(image, detection)
            coords = np.zeros((68, 2))

            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)

            return coords

        except RuntimeError:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        Public intance method that aligns an image for face verification

        Arguments:
         - image is a numpy.ndarray of rank 3 containing
            the image to be aligned
         - landmark_indices is a numpy.ndarray of shape (3,) containing
            the indices of the three landmark points that should be
            used for the affine transformation
         - anchor_points is a numpy.ndarray of shape (3, 2) containing
            the destination points for the affine transformation,
            scaled to the range [0, 1]
         - size is the desired size of the aligned image

        Returns:
         A numpy.ndarray of shape (size, size, 3) containing
         the aligned image, or None if no face is detected
        """

        detected = self.detect(image)
        coords = self.find_landmarks(image, detected)
        in_points = coords[landmark_indices]
        in_points = in_points.astype('float32')
        out_points = anchor_points * size
        warp_mat = cv2.getAffineTransform(in_points, out_points)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

        return warp_dst
