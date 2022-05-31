import  tensorflow as tf 
from tensorflow import keras as keras
import numpy as np
from utils import download_data ,read_yaml
import matplotlib.pyplot as plt


train_data, test_data=download_data()
config= read_yaml("config.yaml")

class Dataloader(tf.keras.utils.Sequence):
    """
    this is dataloader take batch size and data as input 

     return the batch of data
    
    
    """
    def __init__(self,batch_size=config['batch_size'],d_data=train_data):
        
        self.batch_size = batch_size
        self.dataS = d_data


    def __len__(self):
        return int(np.ceil(len(self.dataS) / float(self.batch_size)))


    def __create_shifted_frames(self,data):
        '''
        agu: data --> list of frames each frame is a numpy array and each instance is a list of 20 frames

        this function create the shifted frames
        for next frame prediction the current labele of a input frame will the imidiate next frame
        a single instance of this data is combination of 20 frames
        
        '''
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y
    

    def __getitem__(self, idx):
        x, y = self.__create_shifted_frames(self.dataS)
        batch_x =x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y =y[idx * self.batch_size:(idx + 1) * self.batch_size]


        img=np.zeros((self.batch_size,19,64,64,1),dtype="float32")
        lable=np.zeros((self.batch_size,19,64,64,1),dtype="float32")

        for indx, i in  enumerate(batch_x):
            img[indx]=x[indx]
        
        for indx, i in  enumerate(batch_y):
            lable[indx]=y[indx]

        return img,lable


# if __name__ == '__main__':
#     dataloader = Dataloader(batch_size=32)
#     batch1=dataloader[1]
#     fig, axes = plt.subplots(4, 5, figsize=(10, 8))
#     plt.imshow(batch1[0][1][12][:,:,0])
