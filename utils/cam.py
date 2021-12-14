import numpy as np
import cv2

def return_CAM(feature_conv, weight, class_idx):
    size_upsample = (250, 250)
    
    bz, nc, h, w = feature_conv.shape 
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))
        cam = np.matmul(weight[idx], beforeDot) 
        cam = cam.reshape(h, w) 
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam