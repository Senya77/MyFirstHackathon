from ultralytics import YOLO
from abc import ABC, abstractmethod
import numpy as np
import cv2
import sys


class Detector(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray):
        pass

class YOLOv8(Detector):
    def __init__(self, detector_model_path):
        self.detector_model_path = detector_model_path
        self.model = YOLO(self.detector_model_path)
    
    def predict(self, image):
        result = self.model.predict(image, save=True, imgsz=1280, conf=0.4)
        return result

def main(name):
    yolo_model = YOLOv8(detector_model_path='yolo_model\\model.pt')
    result = yolo_model.predict(name)

if __name__ == '__main__':
    main('')