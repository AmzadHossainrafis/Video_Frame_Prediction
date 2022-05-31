import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Input,ConvLSTM2D,BatchNormalization,Conv3D
from tensorflow.keras.models import Model
from utils import read_yaml

config= read_yaml("config.yaml")

def LSTM2D():
    """
    arguments: input_shape(tuple)
    returns: model(keras.model)
    """
    input_layer = Input(shape=(19,config["height"],config["width"],config["channels"]))
    x = ConvLSTM2D( filters=config['filter_size'],kernel_size=(5, 5),padding=config['padding'],return_sequences=True,activation=config['activation'])(input_layer)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=config['filter_size'],kernel_size=(3, 3),padding=config['padding'],return_sequences=True, activation=config['activation'])(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=config['filter_size'],kernel_size=(1, 1),padding=config['padding'],return_sequences=True,activation=config['activation'])(x)
    x = Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding=config['padding'])(x)

# Next, we will build the complete model and compile it.
    model = Model(input_layer, x)
    return model
