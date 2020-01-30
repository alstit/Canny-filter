'''
Created on 29 janv. 2020
@author: User
'''
from PIL import Image
import numpy



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

    kernel = numpy.empty((0,n-1))
    
    for i in range(-n/2,n/2):
        kernel[i] = 1/(Sigma*sqrt(2*PI)))*exp(-((i-mu)**2)/(2*Sigma**2))

return kernel
        
 

# FoG(x) = Somme for kernel size(F(x)*G(x-i)) 
def convolution(input, kernel):
    output = numpy.empty(input.shape)
    k =0
    for line in input:
        for i in range(0, line.size()):
            for j in range(0,kernel.size())
            outpline[i] += line[(i+kernel.size()/2)-j]*kernel[kernel.size()-j]
        output[k] = outline
            
    return output



    
    
 
if __name__ == '__main__':
    
    img = Image.open("./input/lenna.png", "r")
    img = numpy.array(img)
    
    #print(img)
    
    img = toGrayscale(img)
    
    #Gausskernel
    kernel = Gauss_kernel(3,5)
    
    #convolution
    img = convolution(img,kernel)
    
    #luminance gradiant
    
    #local max
    
    #hysteresis
    


    out=Image.fromarray(img)
    out.show()
    
    
    pass