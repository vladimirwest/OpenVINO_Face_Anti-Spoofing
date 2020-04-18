import cv2
import torchvision
from PIL import Image
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
  
PATH_MODEL_480 = 'anti_spoofing_480'
res = 480

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False): 
    print("Unable to read camera feed")
width = int(cap.get(3))
height = int(cap.get(4))
center = (int(width / 2), int(height / 2))
upper_left = (center[0] - center[1]), 0
bottom_right = (width - (center[0] - center[1])), (height)

spoofNet = cv2.dnn.readNet(PATH_MODEL_480 +'.bin',
                              PATH_MODEL_480 + '.xml')
#spoofNet.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)


data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(res),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

while(True):
    ret, frame = cap.read()
    if ret == True:
        img_cam = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
        cv2_im = cv2.cvtColor(img_cam,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        img = data_transforms(pil_im).cpu().numpy()
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0)
        spoofNet.setInput(img)
        out = spoofNet.forward()
        cv2.putText(img_cam, 'spoof prob: '+str(sigmoid(out[0][0]))[0:5], (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame',img_cam)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break  

cap.release()

cv2.destroyAllWindows()  