import tensorflow as tf 
from tensorflow.keras.layers import LSTM, Conv2D, MaxPooling2D,TimeDistributed, Dropout,Flatten, Dense
from tensorflow.keras.models import Sequential

def make_LRCN_model(SEQUENCE_LENGTH, VIDEO_IMAGE_HEIGHT, VIDEO_IMAGE_WIDTH, CLASSES_LIST):
    '''
    This function will construct the requied LRCN model.
    Returns:
        model: Created LRCN model.
    '''

    # We will use a Sequential model for model construction.
    model = Sequential()
    
    #TimeDistributed layer has been used as dealing with the sequential data,
    #and we want to feed our Conv2D layer to all the the time steps .  
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQUENCE_LENGTH, VIDEO_IMAGE_HEIGHT, VIDEO_IMAGE_WIDTH, 3)))
    
    #MaxPoooling has been used to reduce the size of the input slowly
    model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
                                      
    model.add(TimeDistributed(Flatten()))

    #LSTMs always need input as 3D data, and that's taken care of by the above CNN model                                   
    model.add(LSTM(32))
                                      
    model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

    return model