from ultralytics import YOLO
import os

class YOLOv8():
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
    foldor_name = 'output_orb'
    dir_list = os.listdir(foldor_name)
    for file in dir_list:
        main(f'{foldor_name}\\{file}')