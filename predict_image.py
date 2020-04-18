import argparse
import cv2
import torchvision
from PIL import Image
import numpy as np
import math

parser = argparse.ArgumentParser(description='Run single image face anti-spoofing with OpenVINO')
parser.add_argument('-i', dest='input', help='Path to input image')
args = parser.parse_args()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
  
PATH_MODEL_480 = 'anti_spoofing_480'
res = 480

spoofNet = cv2.dnn.readNet(PATH_MODEL_480 +'.bin',
                              PATH_MODEL_480 + '.xml')
#spoofNet.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)


data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(res),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


input_img = cv2.imread(args.input)
height, width, channels = input_img.shape 
center = (int(width / 2), int(height / 2))
if(width>height):
    upper_left = (center[0] - center[1]), 0
    bottom_right = (height - (center[0] - center[1])), (width)
    input_img = input_img[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]


cv2_im = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(cv2_im)
img = data_transforms(pil_im).cpu().numpy()

img = img.astype(np.float32)
img = np.expand_dims(img, 0)
spoofNet.setInput(img)
out = spoofNet.forward()

print('Spoof probability:' + str(sigmoid(out[0][0]))[0:5])
cv2.putText(input_img, 'Spoof probability: ' + str(sigmoid(out[0][0]))[0:5], (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('Spoof probability', input_img)
cv2.waitKey()