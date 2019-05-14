# !/usr/bin/python
import os
import csv
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import math


def mser(img):
    """ 
    MSER text detection
    """
    mser = cv2.MSER_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting to GrayScale
    gray_img = img.copy()

    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(gray_img, hulls, 1, (0, 0, 255), 2)
    # plt.axis("off")
    # plt.imshow(gray_img)
    # plt.show()
    return hulls

def detect_edge(gray):
    """ 
    Canny edge detection
    """
    edges = cv2.Canny(gray,100,200)
    # plt.axis("off")
    # plt.imshow(edges)
    # plt.show()
    return edges

def color_hist(img):
    """ 
    Plot BGR color histogram
    """
    color = ('b','g','r')
    # plt.axis("off")
    # plt.hist(img.ravel(),256,[0,256]); plt.show()
    for i,col in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color = col)
        plt.xlim([0,256])
    # plt.show()
    return hist

def hsv_hist(img):
    """ 
    Plot HSV histogram
    """
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    # plt.axis("off")
    # plt.imshow(hist,interpolation = 'nearest')
    # plt.show()
    return hist


def find_histogram(clt):
    """
    Create a histogram with k clusters 
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors2(hist, centroids):
    """
    Plot bar charts for dominant colors
    """
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def get_dominant_color(img):
    """
    Get 2 dominant colors in img using 3 clusters
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    # print(clt.cluster_centers_)
    count = Counter(clt.labels_)
    # print(count)
    dominant = [pair[0] for pair in sorted(count.items(), key=lambda item: item[1])]
    # print(dominant[-1], dominant[-2])
    dom1 = clt.cluster_centers_[dominant[-1]]
    dom2 = clt.cluster_centers_[dominant[-2]]
    # print(dom1,dom2)

    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()
    return dom1, dom2

def get_dominant_color2(img):
    """
    Get most dominant colors in img using 3 clusters
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    # print(clt.cluster_centers_)
    count = Counter(clt.labels_)
    # print(count)
    dominant = [pair[0] for pair in sorted(count.items(), key=lambda item: item[1])]
    # print(dominant[1])
    dom1 = clt.cluster_centers_[dominant[1]]
    # print(dom1)

    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()
    return dom1


def seperate_p_f_color(img, peripheral):
    """
    Seperate peripheral and foreground dominant color
    """
    dom1, dom2 = get_dominant_color(img)
    dom_peripheral = get_dominant_color2(peripheral)
    dist1 = cv2.norm(dom_peripheral - dom1)
    dist2 = cv2.norm(dom_peripheral - dom2)
    if dist1 > dist2:
        dom_foreground = dom1
    else:
        dom_foreground = dom2
    return dom_peripheral, dom_foreground


def detect_peripheral_lines(img):
    """
    Detect number of peripheral lines
    """
    gray = cv2.cvtColor(peripheral, cv2.COLOR_BGR2GRAY)
    edges = detect_edge(gray)
    minLineLength = 10
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 90, threshold=10, lines=np.array([]), minLineLength=minLineLength, maxLineGap=2)
    if lines is not None:
        return len(lines)
    else:
        return 0


def rgb2hsv(rgb):
    """
    Helper functoin to convert RGB to HSV
    """
    rgb = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb , cv2.COLOR_RGB2HSV)
    return hsv[0][0]


def get_p_f(img, peripheral):
    """
    Get peripheral and foreground color in both RGB and HSV
    """
    p_rgb, f_rgb = seperate_p_f_color(img,peripheral)
    f_hsv = rgb2hsv(f_rgb)
    p_hsv = rgb2hsv(p_rgb)
    return p_hsv, f_hsv, p_rgb, f_rgb


if __name__ == '__main__':
    # Set the directory you want to start from
    rootDir = '../data'
    char_count_dict = {}
    char_sum = 0
    data_list = []

    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            fname_list = fname.split('_')
            character = fname_list[0]
            if character == '.DS':
                continue
            if character == 'colon': 
                character = ':'
            if character == 'question': 
                character = '?'
            if character == 'dot': 
                character = '.'
            readability = fname_list[1]
            finfo = ['../data/'+fname, character, int(readability)]
            data_list.append(finfo)
            char_sum += 1
            if character in char_count_dict:
                char_count_dict[character] += 1
            else:
                char_count_dict[character] = 1

    # Sort the dictionary by ascii order and plot bar graph
    sorted_keys = sorted(char_count_dict.keys())
    sorted_values = []
    for key in sorted_keys:
        sorted_values.append(char_count_dict[key])

    plt.bar(sorted_keys, sorted_values, color='b')
    plt.show()

    train_data = []
    i = 1
    for finfo in data_list:
        print ("Processing file: ", i, " of 1425")
        i += 1
        train_info = []
        fname = finfo[0]
        
        # Read image and resize
        img = cv2.imread(fname)
        img = cv2.resize(img,(200,200))

        # Mask image
        mask = cv2.imread('mask.png',0)
        peripheral = cv2.bitwise_and(img,img,mask = mask)

        # Process the image
        p_hsv, f_hsv, p_rgb, f_rgb = get_p_f(img, peripheral)
        line_count = detect_peripheral_lines(peripheral)

        finfo.append(p_hsv[0])
        finfo.append(p_hsv[1])
        finfo.append(p_hsv[2])
        finfo.append(f_hsv[0])
        finfo.append(f_hsv[1])
        finfo.append(f_hsv[2])
        finfo.append(p_rgb[0])
        finfo.append(p_rgb[1])
        finfo.append(p_rgb[2])
        finfo.append(f_rgb[0])
        finfo.append(f_rgb[1])
        finfo.append(f_rgb[2])
        finfo.append(line_count)
        train_info.append(p_hsv[0])
        train_info.append(line_count)
        train_data.append(train_info)

    # Save results
    with open('img_info_2.csv', 'w') as csvFile:
        for finfo in data_list:
            writer = csv.writer(csvFile)
            writer.writerow(finfo)
    csvFile.close()

    with open('train_info.csv', 'w') as csvFile:
        for tinfo in train_data:
            writer = csv.writer(csvFile)
            writer.writerow(tinfo)
    csvFile.close()
