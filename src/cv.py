import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import math

def mser(img):
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
    edges = cv2.Canny(gray,100,200)
    # plt.axis("off")
    # plt.imshow(edges)
    # plt.show()
    return edges

def color_hist(img):
    color = ('b','g','r')
    # plt.axis("off")
    # plt.hist(img.ravel(),256,[0,256]); plt.show()
    for i,col in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color = col)
        plt.xlim([0,256])
    # plt.show()
    return hist

def gaussian_blur(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    # plt.axis("off")
    # plt.imshow(blur)
    # plt.show()
    return blur

def hsv_hist(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    # plt.axis("off")
    # plt.imshow(hist,interpolation = 'nearest')
    # plt.show()
    return hist


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors2(hist, centroids):
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

    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    return dom1


def seperate_p_f_color(img, peripheral):
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
    gray = cv2.cvtColor(peripheral, cv2.COLOR_BGR2GRAY)
    edges = detect_edge(gray)
    minLineLength = 10
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 90, threshold=10, lines=np.array([]), minLineLength=minLineLength, maxLineGap=2)
    print(len(lines))
    if lines is not None:
        # for i in range(0, len(lines)):
        #     l = lines[i][0]
        #     cv2.line(peripheral, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        return len(lines)
    else:
        return 0
    # plt.imshow(edges)
    # plt.show()
    # plt.imshow(peripheral)
    # plt.show()


def rgb2hsv(rgb):
    rgb = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb , cv2.COLOR_RGB2HSV)
    return hsv[0][0]


def get_hsv_p_f(img, peripheral):
    p, f = seperate_p_f_color(img,peripheral)
    f_hsv = rgb2hsv(f)
    p_hsv = rgb2hsv(p)
    return p_hsv, f_hsv



if __name__ == "__main__":
    img = cv2.imread('../data/I_2_09.png')
    img = cv2.resize(img,(200,200))
    # plt.imshow(img)
    # plt.show()
    # color_hist(img)
    # mser(img)
    # blur = gaussian_blur(img)
    # detect_edge(blur)
    # hsv_hist(img)
    
    mask = cv2.imread('mask.png',0)
    peripheral = cv2.bitwise_and(img,img,mask = mask)
    # plt.imshow(peripheral)
    # plt.show()

    

