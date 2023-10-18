"""
Implementation of of "FoleyGAN: Visually Guided Generative Adversarial 
Network-Based Synchronous Sound Generation in Silent Videos".

https://arxiv.org/pdf/2107.09262.pdf
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio.transforms as audiotransforms
from models.TRNmodule import RelationModule, RelationModuleMultiScale
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)


# What is img_feature_dim?
class foleygan(nn.Module):
    def __init__(
        self, 
        is_grayscale: bool = True,
        img_feature_dim, 
        num_frames, 
        num_class
        ):
        super(foleygan, self).__init__()
        self.grayscale_adapter = nn.Linear(1, 3) if is_grayscale else None
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1, num_classes=num_class)

        self.trn = RelationModule(img_feature_dim, num_frames, num_class)
        self.mtrn = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)

        self.spectrogram = audiotransforms.Spectrogram()

        self.biggan = BigGAN.from_pretrained('biggan-deep-512')

def forward(self, x):
        # Apply ResNet50
        x_resnet50 = self.cnn(x)

        x_mtrn = self.mtrn(x_resnet50)
        x_trn = self.trn(x_resnet50)

        x_spectrogram = self.spectrogram(x_trn)
        #x_class = self.fc1(x_mtrn)
        x_class = x_mtrn

        x = self.biggan(x_spectrogram, x_class)
        return x

# class LRCNModel(nn.Module):
#     #def __init__(self, sequence_length, video_image_height, video_image_width, classes_list=10):
#     def __init__(self, num_classes = 15):
#         super(LRCNModel, self).__init__()
        
#         # Create the CNN layers with TimeDistributed
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=(3, 3), padding='same'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(4, 4)),
#             nn.Dropout(0.25),
            
#             nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(4, 4)),
#             nn.Dropout(0.25),
            
#             nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2)),
#             nn.Dropout(0.25),
            
#             nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
        
#         # LSTM layer
#         self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        
#         # Fully connected layer
#         self.fc = nn.Linear(32, num_classes)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         # Apply CNN layers with TimeDistributed
#         x = self.cnn(x)
        
#         # Reshape data for LSTM (incoming x is (batchsize, num_frames, height, width))
#         x = x.view(x.shape[0], x.shape[1], -1)
        
#         # Apply LSTM layer
#         x, _ = self.lstm(x)
        
#         # Get the output of the last time step
#         x = x[:, -1, :]
        
#         # Apply fully connected layer and softmax
#         x = self.fc(x)
#         x = self.softmax(x)
        
#         return x


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