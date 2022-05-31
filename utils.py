from tensorflow import keras
import numpy as np
import yaml
import math
import os 
import  matplotlib.pyplot as plt

#load model 
from keras.models import load_model


def download_data(link="http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",normalize=True,number_of_data=1000,split_ratio=0.8):
    """
    Downloads the data from the link provided.
    :param link: The link to download the data from.
    :return: The data.


    """
    dataset=keras.utils.get_file("mnist_test_seq.npy", link)
    dataset=np.load(dataset)
    dataset = np.swapaxes(dataset, 0, 1) # separet all the frame(single image) in the sequence of frams(multiple images)
    dataset=dataset[:number_of_data,...]
    dataset = np.expand_dims(dataset, axis=-1) # Add the channel dimension. as it is a grayscale image.
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(split_ratio * dataset.shape[0])]
    val_index = indexes[int(split_ratio * dataset.shape[0]) :]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    if normalize:
        train_dataset = train_dataset / 255
        val_dataset = val_dataset / 255

    
    return train_dataset,val_dataset

    #(800, 20, 64, 64, 1)
    #(200, 20, 64, 64, 1)

def read_yaml(path='config.yaml'):
    """
    Reads the yaml file and returns the data in a dictionary.
    :param path: The path to the yaml file.
    :return: The data in the yaml file.
    """
    with open(path, 'r') as stream:
        data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
    return data_loaded



class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self,config= read_yaml()):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
            config (dict): configuration dictionary
        Return:
            class object
        """



        super(keras.callbacks.Callback, self).__init__()
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """


        drop = 0.5
        epoch_drop = self.config['epochs'] / 8.
        lr = self.config['learning_rate'] * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr


    def get_callbacks(self):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """


        
        if self.config['csv']:
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(self.config['csv_log_dir'], self.config['csv_log_name']), separator = ",", append = False))
        if self.config['checkpoint']:
            self.callbacks.append(keras.callbacks.ModelCheckpoint(filepath=self.config['checkpoint_dir']+"next_frame_prediction.hdf5", save_best_only = True))
        if self.config['lr']:
            self.callbacks.append(keras.callbacks.LearningRateScheduler(schedule = self.lr_scheduler))
        
        
        
        return self.callbacks



def plot_loss(history):
    """
    Summary:
        plot the loss function
    Arguments:
        history (object): keras.callbacks.History object
    Return:
        None
    """


    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","test"],loc="upper left")
    plt.show()
    plt.savefig("data\prediction\accuracy.png")    




def predictions(model_dir,datasets):

    """
    arg: model_dir --> str , contain the pathe of model dataset
    arg: datasets --> list , contain the dataset for prediction

    return: list of predictions and polt it 
    
    """
    model=load_model(model_dir)
    fig ,axis= plt.subplot(2,5)
    rand=np.random.randint(0,14)
    for i in range(2):
        for j in range(5): 
            rand_frame=datasets[0,rand:,...]
            img=rand_frame[:,:,0]
            img=np.expand_dims(rand_frame+j,axis=0)
            pred=model.predict(img)
            prediction=pred[0:0,:,:,0]
            axis[i][j].plt.imshow(prediction, cmap='gray')





        



