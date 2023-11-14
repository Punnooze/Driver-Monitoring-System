import uuid
import os
import time
import cv2

IMAGES_PATH = os.path.join('data','images')
labels = ['phone']
number_imgs=30


cap = cv2.VideoCapture(0)
for label in labels:
    print('Collecting Images for {}'.format(labels))
    time.sleep(5)

    for img_num in range(number_imgs):
        print('Collecting images for {}, image number{}'.format(label,img_num))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        cv2.imwrite(imgname,frame)
        cv2.imshow('Image Collection',frame)
        time.sleep(2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()




