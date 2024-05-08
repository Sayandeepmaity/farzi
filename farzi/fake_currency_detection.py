import cv2
import matplotlib.pyplot as plt
import numpy as np

A = cv2.imread('real.jpg')
P = cv2.imread('fake.jpg')

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB)), plt.title('Real Currency')
plt.subplot(122), plt.imshow(cv2.cvtColor(P, cv2.COLOR_BGR2RGB)), plt.title('Fake Currency')
plt.show()

a = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
p = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(a, cmap='gray'), plt.title('Real Currency (Grayscale)')
plt.subplot(122), plt.imshow(p, cmap='gray'), plt.title('Fake Currency (Grayscale)')
plt.show()

a2tr = a[330:1200, 1016:1927]
b2tr = p[170:1040, 716:1627]

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(a2tr, cmap='gray'), plt.title('Real Currency ROI')
plt.subplot(122), plt.imshow(b2tr, cmap='gray'), plt.title('Fake Currency ROI')
plt.show()

a2_str = a[5:1100, 2080:2151]
p2_str = p[5:1100, 1666:1729]

hsvImageReal = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
hsvImageFake = cv2.cvtColor(P, cv2.COLOR_BGR2HSV)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(hsvImageReal, cv2.COLOR_HSV2RGB)), plt.title('Real Currency (HSV)')
plt.subplot(122), plt.imshow(cv2.cvtColor(hsvImageFake, cv2.COLOR_HSV2RGB)), plt.title('Fake Currency (HSV)')
plt.show()

croppedImageReal = hsvImageReal[5:1100, 2080:2151]
croppedImageFake = hsvImageFake[5:1100, 1666:1729]

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(croppedImageReal, cv2.COLOR_HSV2RGB)), plt.title('Real Currency Cropped (HSV)')
plt.subplot(122), plt.imshow(cv2.cvtColor(croppedImageFake, cv2.COLOR_HSV2RGB)), plt.title('Fake Currency Cropped (HSV)')
plt.show()

satThresh = 0.3
valThresh = 0.9

g = croppedImageReal[:,:,1] > satThresh
h = croppedImageReal[:,:,2] < valThresh

g1 = croppedImageFake[:,:,1] > satThresh
h1 = croppedImageFake[:,:,2] < valThresh

BWImageReal = g & h
BWImageFake = g1 & h1

def bwareaopen(img, min_size, connectivity=8):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]
        if label_size < min_size:
            img[labels == i] = 0
    return img

binr = cv2.threshold(a2_str, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
invert = cv2.bitwise_not(binr)
BWImageCloseReal = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

binr2 = cv2.threshold(p2_str, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
invert2 = cv2.bitwise_not(binr2)
BWImageCloseFake = cv2.morphologyEx(invert2, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

areaopenReal = bwareaopen(BWImageCloseReal, 15)
areaopenFake = bwareaopen(BWImageCloseFake, 15)

countReal = cv2.connectedComponentsWithStats(areaopenReal, 8)
countFake = cv2.connectedComponentsWithStats(areaopenFake, 8)

def corr2(A, B):
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

co = corr2(a2tr, b2tr)

if (co.any() >= 0.5):
    print('Correlation of transparent area > 0.5')
    if (countReal[0] == countFake[0]):
        print('Currency is legitimate')
    else:
        print('Green strip is fake')
else:
    print('Correlation of transparent area < 0.5')
    print('Currency is fake')
