# -*- coding: utf-8 -*-

###############################################################
###                                                         ###
###    Data Augmentation Generator Example                  ###
###                                                         ###
### About: This example takes the original images in the    ###
###         train and validation class label named          ###
###         sub-directories then creates many permuted      ###
###         images based on the chosen parameters.          ###
###         Basically, this allows you to go from 197       ###
###         images in 3 classes to about 5000 images in 3   ###
###         classes.                                        ###
###                                                         ###
###         Also, this program runs the code effectively    ###
###         twice. This is to generate different images     ###
###         for training and validation sets.               ###
###                                                         ###
### Created by: Chris Havenstein                            ###
###                                                         ###
### Last Modified on 8/12/2019                              ###
###                                                         ###
###############################################################

#import the required libraries
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import time

#start a timer.
start = time.time()


#for examples of data augmentation possibilities, check out: 

#first: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
#second: https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
#third: https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
#fourth: https://towardsdatascience.com/image-augmentation-for-deep-learning-using-keras-and-histogram-equalization-9329f6ae5085


# First, create our ImageDataGenerator which sets the possibilities for how to create additional images
# https://keras.io/preprocessing/image/
datagen = ImageDataGenerator(
        rotation_range=40,      #rotate the image randomly between 0 to 40%
        width_shift_range=0.2,  #shift by the width with possible values are floats in the half-open interval [-0.1,+0.1]
        height_shift_range=0.2, ##shift by the height with possible values are floats in the half-open interval [-0.1,+0.1]
        shear_range=0.2,        #shear intensity, Shear angle in counter-clockwise direction in degrees.
        zoom_range=0.2,         #range for random zoom, [1-zoom_range, 1+zoom_range].
        horizontal_flip=True,   #You can horizontally flip the image.
        #vertical_flip=True,    #You can vertically flip the image (upside-down)
        fill_mode='nearest')    #fill pixels based on the nearest pixel value.

#specifiy your relative directories for your class images in your working directory.
cat_dir = './train/cats/'
dog_dir = './train/dogs/'
horse_dir = './train/horses/'

#First do it to get new training images.

#For each image in the cat directory, cat_dir
for filename in os.listdir(cat_dir):
    #If the image ends with .jpg
    if filename.endswith(".jpg"): 
    
        
        filename = cat_dir + str(filename) # convert the cat_dir string and filename into a concatenated string
        img = load_img(filename)  # this creates a PIL image
        x = img_to_array(img)  # this is a Numpy array created from the PIL image
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with the correct shape for datagen
            
        i = 0
        #For each image in the cat_dir, we create 30 new images and save them in the train2/cats folder        
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='train2/cats', save_prefix='cat', save_format='jpeg'):
            i += 1
            if i > 30:
                break
            
#For each image in the dog directory, dog_dir
for filename in os.listdir(dog_dir):
    #If the image ends with .jpg
    if filename.endswith(".jpg"): 
             
        filename = dog_dir + str(filename) # convert the dog_dir string and filename into a concatenated string
        img = load_img(filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with the correct shape for datagen        
        i = 0

        #For each image in the dog_dir, we create 30 new images and save them in the train2/dogs folder        
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='train2/dogs', save_prefix='dog', save_format='jpeg'):
            i += 1
            if i > 30:
                break  

#For each image in the horse directory, horse_dir
for filename in os.listdir(horse_dir):
    #If the image ends with .jpg
    if filename.endswith(".jpg"): 
             
        filename = horse_dir + str(filename) # convert the horse_dir string and filename into a concatenated string
        img = load_img(filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array 
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with the correct shape for datagen
        
        i = 0
        
        #For each image in the horse_dir, we create 30 new images and save them in the train2/horses folder
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='train2/horses', save_prefix='horse', save_format='jpeg'):
            i += 1
            if i > 30:
                break        

#Now do it again to get new validation images.
                
#For each image in the cat directory, cat_dir
for filename in os.listdir(cat_dir):
    #If the image ends with .jpg
    if filename.endswith(".jpg"): 
    
        
        filename = cat_dir + str(filename) # convert the cat_dir string and filename into a concatenated string
        img = load_img(filename)  # this creates a PIL image
        x = img_to_array(img)  # this is a Numpy array created from the PIL image
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with the correct shape for datagen
            
        i = 0
        #For each image in the cat_dir, we create 30 new images and save them in the validation2/cats folder        
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='validation2/cats', save_prefix='cat', save_format='jpeg'):
            i += 1
            if i > 30:
                break
            
#For each image in the dog directory, dog_dir
for filename in os.listdir(dog_dir):
    #If the image ends with .jpg
    if filename.endswith(".jpg"): 
             
        filename = dog_dir + str(filename) # convert the dog_dir string and filename into a concatenated string
        img = load_img(filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with the correct shape for datagen        
        i = 0

        #For each image in the dog_dir, we create 30 new images and save them in the validation2/dogs folder        
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='validation2/dogs', save_prefix='dog', save_format='jpeg'):
            i += 1
            if i > 30:
                break  

#For each image in the horse directory, horse_dir
for filename in os.listdir(horse_dir):
    #If the image ends with .jpg
    if filename.endswith(".jpg"): 
             
        filename = horse_dir + str(filename) # convert the horse_dir string and filename into a concatenated string
        img = load_img(filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array 
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with the correct shape for datagen
        
        i = 0
        
        #For each image in the horse_dir, we create 30 new images and save them in the validation2/horses folder
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='validation2/horses', save_prefix='horse', save_format='jpeg'):
            i += 1
            if i > 30:
                break         
      
#find the ending time...
end = time.time()

#print the total runtime.
print(end - start)