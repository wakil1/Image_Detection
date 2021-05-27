import cv2
import os
import math

images= []
classN = []
List = []
myList=os.listdir('images')

print(myList)

siftDetection =cv2.SIFT_create(nfeatures=1500)
imagesFolder ='images'
threshold= 51

# you can change thres depending on how accurate you want it to be
for attributes in myList:
    imgC = cv2.imread(f'{imagesFolder}/{attributes}',0)
    images.append(imgC)
    classN.append(os.path.splitext(attributes)[0])

print(classN)
#detectfeatures for each image

for imges in images:

    kp, des = siftDetection.detectAndCompute(imges, None)
    List.append(des)

video =cv2.VideoCapture(0)


def importVideoImage():
    success, newImage = video.read()
    imgOrginal = newImage.copy()
    # convert image to gray
    newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    kp2, des2 = siftDetection.detectAndCompute(newImage, None)
    matching = []
    finalVal = -15
    bf = cv2.BFMatcher()
    for ds1 in List:
        matchDetection = bf.knnMatch(ds1, des2, k=2)
        goodDetector = []

        for i, j in matchDetection:

            if i.distance > 0.77 * j.distance:
                pass
            else:
                goodDetector.append([i])

        matching.append(len(goodDetector))
    print(matching)

    if max(matching) > threshold:
        finalVal = matching.index(max(matching))

    if finalVal == -15:
        pass;
    else:
        cv2.putText(imgOrginal, classN[finalVal], (75, 75), cv2.Formatter_FMT_NUMPY, 1, (255, 255, 0), 1)
    cv2.imshow('newImage', imgOrginal)
    cv2.waitKey(1)



while True:
    importVideoImage()


