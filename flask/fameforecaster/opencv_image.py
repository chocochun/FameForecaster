from io import StringIO as sio
from PIL import Image
import urllib
import scipy as sp
import scipy.cluster as sc
import numpy as np
import colorsys
from PIL import ImageStat
import cv2
import pandas as pd

class Img_feature:
    
    def __init__(self,image):
        self.imagepath = image
        self.myClusters = 5
        
        # initialize color
        self.colurlist = ['blacks', 'blues', 'cyans', 'grays', 'greens', 'magentas',
       'reds', 'whites', 'yellows'] + ['brightness', 'ratioUniqueColors',
                               'thresholdBlackPerc', 'highbrightnessPerc','lowbrightnessPerc',
                              'CornerPer','EdgePer','FaceCount']
        for i in range(0, len(self.colurlist)):
            setattr(self, self.colurlist[i], 0)                

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        
    def get_vector(self):
        
        self.img = Image.open(self.imagepath)
        self.img_array = np.asarray(self.img)
        self.vector = self.img_array.reshape(sp.product(self.img_array.shape[:2]), self.img_array.shape[2]).astype(float)

    def meanBrightness(self):
        ### colour image to grasyscale
        ### uses the ITU-R 601-2 luma transform
        ### full list of modes at http://effbot.org/imagingbook/concepts.htm#mode
        myImageBW = self.img.convert('L')

        ### this will essentially give a histogram of grayscale values from 0 to 255
        myImageStat = ImageStat.Stat(myImageBW)
        self.brightness = int(myImageStat.mean[0])
        
    def get_DominantColor(self):
        # based on http://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
        ### get clusters from array
        codes, distortion = sc.vq.kmeans(self.vector, self.myClusters)

        ### Assign codes to the array
        vecs, dist = sc.vq.vq(self.vector, codes) 

        ### Count occurances of each code
        counts, bins = sp.histogram(vecs, len(codes))

        ### find which index has the max value
        index_max = sp.argmax(counts)   

        # get that color
        peak = codes[index_max]

        ### ### HLS (Hue Lightness Saturation) and HSV (Hue Saturation Value).
        HLS = colorsys.rgb_to_hls(peak[0]/255.0,peak[1]/255.0,peak[2]/255.0)

        #print myDominantColorRGB[i]
        hueAngle = HLS[0]*360

        # Lightness
        if (HLS[1] < 0.2):
            self.blacks = 1
            self.DominantColor = "Black"
            
        elif (HLS[1] > 0.8):
            self.whites = 1
            self.DominantColor = "White"
            
        # saturation
        elif (HLS[2] < 0.25):
            self.grays = 1
            self.DominantColor = "Gray"
            
        # hue
        elif (hueAngle < 30):
            self.reds = 1
            self.DominantColor = "Red"
            
        elif (hueAngle < 90):
            self.yellows = 1
            self.DominantColor = "Yellows"
            
        elif (hueAngle < 150):
            self.greens = 1
            self.DominantColor = "Green"
            
        elif (hueAngle < 210):
            self.cyans = 1
            self.DominantColor = "Cyan"
            
        elif (hueAngle < 270):
            self.blues = 1
            self.DominantColor = "Blue"
            
        elif (hueAngle < 330):
            self.magentas = 1
            self.DominantColor = "Magenta"

        else:
            self.reds = 1
            self.DominantColor = "Red"

    def uniqueColorRatio(self):
        ### get number of unique colors (the more, the colorful)
        # take image as is - do not convert to grayscale
        myImageStatColor = ImageStat.Stat(self.img) 
        myList = self.vector.tolist()
        myUnique = [list(x) for x in set(tuple(x) for x in myList)]
        uniqueColors = len(myUnique) 
        # now to get percentage (no of unique pixels/count of all pixels)
        # if 1 image is super colorful, if close to zero, then there is a handful of colors only
        # if zero, we know it is grayscale not color
        self.ratioUniqueColors = round(float(uniqueColors)/myImageStatColor.count[0],2)

    def get_grey(self):
        self.grey = cv2.cvtColor(self.img_array,cv2.COLOR_BGR2GRAY)
        self.grey = np.float32(self.grey)

        
    def blackP(self):
        ret,th1 = cv2.threshold(self.grey,127,255,cv2.THRESH_BINARY)
        black=0;
        for i in th1:
            for j in i:
                if (j==0):
                    black+=1
        rows,cols,depth= self.img_array.shape
        blackPerc=black*100/float(rows*cols)
        blackPerc= "%.2f" % (blackPerc) 
        self.thresholdBlackPerc=float(blackPerc)

    def brightP(self):
        Lum=[]
        for i in self.img_array:
            for R,G,B in i:
                Lum.append((0.2126*R) + (0.7152*G) + (0.0722*B))  
        lm=np.mean(Lum)
        maxlm=[i for i in Lum if i>2*lm]
        minlm=[i for i in Lum if i<0.5*lm]
        maxlmP= len(maxlm)*100/float(len(Lum))
        maxlmP= "%.2f" % (maxlmP) 
        self.highbrightnessPerc=float(maxlmP)
        minlmP= len(minlm)*100/float(len(Lum))
        minlmP= "%.2f" % (minlmP) 
        self.lowbrightnessPerc=float(minlmP)
        
    def cornerP(self):
        dst = cv2.cornerHarris(self.grey,2,3,0.04)
        maxd=dst.max()*0.02
        cornerlist=0
        suml=0
        #cornerlist=[i for i in np.ndenumerate(dst) if i>maxd]
        unzip_lst = zip(*dst)
        for i in unzip_lst:
            for j in i:
                suml+=1
                if j>maxd:
                    cornerlist+=1
        cornerPerc=cornerlist*100/float(suml)
        cornerPerc= "%.2f" % (cornerPerc) 
        self.CornerPer=float(cornerPerc)

    def edgeP(self):
        Edge=0;
        edges = cv2.Canny(self.img_array,100,200) 
        for i in edges:
            for j in i:
                if (j==255):
                    Edge+=1
        rows,cols,depth= self.img_array.shape
        edgePerc=Edge*100/float(rows*cols)
        edgePerc= "%.2f" % (edgePerc) 
        self.EdgePer=float(edgePerc)
        
    def faceC(self):
        gray = cv2.cvtColor(self.img_array,cv2.COLOR_BGR2GRAY) 
        self.FaceCount = len(self.face_cascade.detectMultiScale(gray, 1.3, 5) )

    def get_all_feature(self):
        self.get_vector()
        self.get_grey()
        self.faceC()
        self.edgeP()
        self.cornerP()
        self.brightP()
        self.blackP()
        self.uniqueColorRatio()
        self.get_DominantColor()
        self.meanBrightness()
        
        self.result = pd.DataFrame([getattr(self,i) for i in self.colurlist] )
        self.result = self.result.transpose()
        self.result.columns = self.colurlist
        

