'''
Created on 29 janv. 2020
@author: User
'''
from PIL import Image
import numpy
import math
import matplotlib.pyplot as plt
from numpy import uint



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


# F(x) = (1/(Sigma*sqrt(2*PI)))*exp(-((x-mu)**2)/(2*Sigma**2))
def Gauss_kernel(n,Sigma):

    #kernel = numpy.empty((n,0))
    kernel = []
    for i in range(int(-n/2),int(n/2)+1):
        kernel.append((1/(Sigma*math.sqrt(2*3.14)))*math.exp(-((i)**2)/(2*Sigma**2)))
        #kernel.append(1/(Sigma))
        #print(kernel)
    return kernel
        
 

# FoG(x) = Somme for kernel size(F(x)*G(x-i)) 
def convolutionX(input, kernel):
    output = numpy.empty(input.shape)

    #print(len(outputline))
    k =0
    for line in input:
        outputline = numpy.zeros((line.shape[0],))
        for i in range(0, input.shape[1]):
            for j in range(0,len(kernel)):
                if uint((i+len(kernel)/2)-j) < 0 :
                    break
                if uint((i+len(kernel)/2)-j) >= len(line):
                    break
                outputline[i] += int(line[uint((i+len(kernel)/2)-j)]*kernel[uint(len(kernel)-j-1)])

        output[k] = outputline
        k+=1

            
    return output

def convolutionY(input, kernel):
    output = numpy.empty((input.shape[1],input.shape[0]))

    #print(len(outputline))
    k =0
    for line in input.T:
        outputline = numpy.zeros((line.shape[0],))
        for i in range(0, input.shape[0]):
            for j in range(0,len(kernel)):
                if uint((i+len(kernel)/2)-j) < 0 :
                    break
                if uint((i+len(kernel)/2)-j) >= len(line):
                    break
                outputline[i] += int(line[uint((i+len(kernel)/2)-j)]*kernel[uint(len(kernel)-j-1)])

        output[k] = outputline
        k+=1

            
    return output.T


    
    
 
if __name__ == '__main__':
    
    img = Image.open("./input/lenna.png", "r")
    img = numpy.array(img)
    
    #print(img)
    
    img = toGrayscale(img)
    
    #Gausskernel
    kernel = Gauss_kernel(6,0.3)

    #noise suppresion
    img = convolutionX(img,kernel)
    img = convolutionY(img,kernel)
    
    #luminance gradiant
    kernely = numpy.array([1,0,-1])
    kernelx = numpy.array([1,2,1])
    
    img = convolutionY(img, kernelx)
    img = convolutionX(img, kernely)

    
    kernelx = numpy.array([1,2,1])
    kernely = numpy.array([1,0,-1])    
    
    img = convolutionX(img, kernelx)
    img = convolutionY(img, kernely)

    
    #local max
    
    #hysteresis
    
    #print(img)

    out=Image.fromarray(img)
    out.show()
    
    
    pass