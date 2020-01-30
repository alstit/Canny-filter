'''
Created on 29 janv. 2020

@author: User
'''
from PIL import Image
import numpy
from PIL._imaging import outline


def toGrayscale(img):
    
    #convert to grayscale : Y = 0.299R + 0.587G + 0.114B
    outputimg=numpy.empty((img.shape[0],img.shape[1]))
    i=0
    for line in img:
        outputline = numpy.empty((img.shape[1]))
        j=0
        for pixel in line :
            pixel = int((0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]))
            outputline[j]=(pixel)
            j+=1 
        outputimg[i] = outputline
    
        i+=1
    return outputimg

if __name__ == '__main__':
    
    img = Image.open("./input/lenna.png", "r")
    img = numpy.array(img)
    
    #print(img)
    
    img = toGrayscale(img)
    
    #Gausskernel
    
    #convolution
    
    #luminance gradiant
    
    #local max
    
    #hysteresis
    


    out=Image.fromarray(img)
    out.show()
    
    
    pass