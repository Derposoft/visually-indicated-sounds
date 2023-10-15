"""
Implementation of of "FoleyGAN: Visually Guided Generative Adversarial 
Network-Based Synchronous Sound Generation in Silent Videos".

https://arxiv.org/pdf/2107.09262.pdf
"""

import torch
import torch.nn as nn

class LRCNModel(nn.Module):
    #def __init__(self, sequence_length, video_image_height, video_image_width, classes_list=10):
    def __init__(self, classes_list=["a", "b", "c", "d", "e"]):
        super(LRCNModel, self).__init__()
        
        # Create the CNN layers with TimeDistributed
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),
            nn.Dropout(0.25),
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(32, len(classes_list))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Apply CNN layers with TimeDistributed
        x = self.cnn(x)
        
        # Reshape data for LSTM
        #x = x.view(x.size(0), sequence_length, -1)
        
        # Apply LSTM layer
        x, _ = self.lstm(x)
        
        # Get the output of the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layer and softmax
        x = self.fc(x)
        x = self.softmax(x)
        
        return x







# import tensorflow as tf 
# from tensorflow.keras.layers import LSTM, Conv2D, MaxPooling2D,TimeDistributed, Dropout,Flatten, Dense
# from tensorflow.keras.models import Sequential

# def make_LRCN_model(SEQUENCE_LENGTH, VIDEO_IMAGE_HEIGHT, VIDEO_IMAGE_WIDTH, CLASSES_LIST):
#     '''
#     This function will construct the requied LRCN model.
#     Returns:
#         model: Created LRCN model.
#     '''

#     # We will use a Sequential model for model construction.
#     model = Sequential()
    
#     #TimeDistributed layer has been used as dealing with the sequential data,
#     #and we want to feed our Conv2D layer to all the the time steps .  
#     model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
#                               input_shape = (SEQUENCE_LENGTH, VIDEO_IMAGE_HEIGHT, VIDEO_IMAGE_WIDTH, 3)))
    
#     #MaxPoooling has been used to reduce the size of the input slowly
#     model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
#     model.add(TimeDistributed(Dropout(0.25)))
    
#     model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
#     model.add(TimeDistributed(MaxPooling2D((4, 4))))
#     model.add(TimeDistributed(Dropout(0.25)))
    
#     model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2))))
#     model.add(TimeDistributed(Dropout(0.25)))
    
#     model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2))))
                                      
#     model.add(TimeDistributed(Flatten()))

#     #LSTMs always need input as 3D data, and that's taken care of by the above CNN model                                   
#     model.add(LSTM(32))
                                      
#     model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

#     return model