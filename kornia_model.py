from PIL import Image
import kornia as K
from kornia.contrib import ImageStitcher
import kornia.feature as KF
import torch
import torchvision
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def inference(file_1, file_2): 
    
    image_1 = torchvision.transforms.PILToTensor()(Image.open(file_1)).float().unsqueeze(dim=0)/255
    image_2 = torchvision.transforms.PILToTensor()(Image.open(file_2)).float().unsqueeze(dim=0)/255
    
    IS = ImageStitcher(KF.LoFTR(pretrained='outdoor'), estimator='ransac')
    with torch.no_grad():
        result = IS(image_1, image_2)
        
    return K.tensor_to_image(result)


def main(file1, file2, name):

    res = inference(file1, file2)
    Image.fromarray((res * 255).astype(np.uint8)).save(f'{name}.jpg')


if __name__ == '__main__':
    main('dataset\\1.jpg', 'dataset\\2.jpg', 'test')