from dataloader import Dataloader
from utils import download_data,read_yaml,SelectCallbacks,plot_loss

import matplotlib.pyplot as plt
from model import LSTM2D
import datetime as dt
import os                                                    
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#read config
config= read_yaml("config.yaml")
#load data 
train_data, val_data=download_data()


#make train and val dataloader

train_ds = Dataloader(d_data=train_data,batch_size=config['batch_size'])
val_ds = Dataloader(d_data=val_data,batch_size=config['batch_size'])

#make model 
model=LSTM2D()
#make callbacks
callback=SelectCallbacks()

model.compile(optimizer=config['optimizer'],loss=config['loss'],metrics=["accuracy"])
history1=model.fit(train_ds,epochs=config['epochs'],validation_data=val_ds,callbacks=callback,shuffle=True)


plot_loss(history=history1)

