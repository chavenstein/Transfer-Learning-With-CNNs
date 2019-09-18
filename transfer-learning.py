# coding: utf-8


###############################################################
###                                                         ###
###    Transfer Learning without Data Augmentation Example  ###
###                                                         ###
### About: This example takes the original images from      ###
###         the original images in the train and            ###
###         validation class label named sub-directories    ###
###         and applies transfer learning with the          ###
###         MobileNet model trained on ImageNet.            ###
###                                                         ###
### Created by: Chris Havenstein                            ###
###                                                         ###
### Last Modified on 8/12/2019                              ###
###                                                         ###
###############################################################

# In[1]:


##These are some of the references I used to create this code.

#From https://github.com/aditya9898/transfer-learning
#From https://towardsdatascience.com/transfer-learning-for-image-classification-using-keras-c47ccf09c8c8


#import the following:
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


# In[2]:

#choose any pretained model in: https://keras.io/applications/

# This example uses the MobileNet model trained on imagenet image data
base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

# take the original model output..
x=base_model.output

# then add a layer for pooling
x=GlobalAveragePooling2D()(x)

# And add fully connected network layers (multi-layer perceptron style)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3

# Finally, map to the number of image classes we have (in our case 3, cat, dog, and horse)
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation


# In[3]:


#now we have a model based on our architecture.
#specify the inputs for the model and
#specify the outputs
model=Model(inputs=base_model.input,outputs=preds)


# In[4]:


#we freeze the first 20 layers
#these layers are NOT trainable
for layer in model.layers[:20]:
    layer.trainable=False

#However...
#we retrain the layers after 20 (This is the transfer learning piece)   
for layer in model.layers[20:]:
    layer.trainable=True


# In[5]:


#designate the batch size for mini-batching during gradient descent .. this will be 32
# ... to update the error/loss after each batch through back-propagation.
batch_size = 32   
    
#create the image data generator  first
# We preprocess the data in the way in the way the MobileNet model expects.
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('./train/', # this is where you specify the relative path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)

# In[6]:
#do the same from our validation image data generator
validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

validation_generator=validation_datagen.flow_from_directory('./validation/', # this is where you specify the relative path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)
# In[7]:

#we could also record the bottleneck features per keras' blog
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

#compile the model next
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

#Set up the train and validation step sizes
#Basically, the total number of images across all classes in each generator divided by the chosen batch size.
step_size_train=train_generator.n//train_generator.batch_size
step_size_validation=validation_generator.n//validation_generator.batch_size


#For plotting training versus validation accuracy
filepath="./checkpoints/" + "MobileNet" + "_model_weights.h5"

#This is to plot later, we're creating a checkpoint list, AKA [checkpoint]
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

#for validation generator advice I went here.
#https://github.com/keras-team/keras/issues/2702

#fit the model, 
#  we freezed the first 20 layers
#  everything after the last 20 layers we're retraining
history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data=validation_generator,
                   validation_steps=step_size_validation,
                   epochs=20,
                   callbacks=callbacks_list)

# In[8]:

# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    #loss = history.history['loss']
    #val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    #plt.plot(epochs, val_loss, 'r-')
    #plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')

# In[9]:

#plot the fitting results.
plot_training(history)

# In[10]:

#If you had test data
#print("[INFO] evaluating network...")
#testGen.reset()
#predIdxs = model.predict_generator(testGen,
#	steps=(totalTest // BS) + 1)
 
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
#predIdxs = np.argmax(predIdxs, axis=1)

# save model and architecture to single file
model.save("model_w_no_data_augmentation.h5")
print("Saved model to disk")

# save the learned model weights to a single file.
model.weights("model_w_no_data_augmentation_weights.h5")

#Later, assuming you wanted to reload the model for predictions
#model = loadmodel("model_w_data_augmentation.h5")
#model.load_weights("model_w_data_augmentation_weights.h5")
