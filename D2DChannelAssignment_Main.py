# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:14:00 2020

@author: Yong Zeng
Sub-band assignment for D2D communications
There are K D2D pairs, and N available sub-bands, N<K
Assign one sub-band to each D2D pair so that the sum rate is maximized
For simplicity, no power control is considered and each D2D transmitter sends with a given power P
Three schemes are considered:
    1. Perfect CSI-based channel assignment: Assuming that the perfect CSI of all the K^2*N communication channels are available, including the channel gains of both the desired links and all cross links
    2. Channel gain map (CGM)-based channel assignment: The system maintains a CGM, with the locations of the Tx and rx as the input, and the predicted channel gain h_kj[n] as the output
    Input dimension: 4, (tx-rx coordinates)
    Output dimension: N (number of sub-bands)
    The CGM is constructed and trained using deep neural network (DNN)
    3. Fitted-PLModel-based channel assignment: Path loss model based, and the modelling parameters are obtained by curve fitting based on the training data

The channel assignment problem is combinatorial, which is NP hard
Exhaustive search requires exponential complexity N^K
We use greedy algorithm to sequentially assign channel to each D2D pair 
For the D2D pair under consideration, select the channel that leads to the maximum sum-rate for all D2D pairs considered so far
The result of such a greedy algorithm is affected by the consideration order of the K D2D pairs. There are in total K! possible orders, and we consider a given number of orders and choose the best one
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from keras.layers import Input, Dense
from keras.models import Model
from scipy.interpolate import griddata
from matplotlib import cm
from sklearn.utils import shuffle

GAIN_MIN_dB=-170.0
GAIN_MAX_dB=-50.0

fc_MHz=np.array([824.0,834,880,890,1710,1720,1850.0,1860,1920,1930,2500,2510],dtype="float64")
fc=fc_MHz*10**6
lambda_vec=3.0*10**8/fc

noise_power_density_dBm_Hz=-174.0
BW=1.0*10**6
noise_power_mW=10**(noise_power_density_dBm_Hz/10)*BW
noise_power=noise_power_mW*10**(-3)

Pt_dBm_vec=np.array([-10,0,10.0,20,30])
Pt_mW_vec=10**(Pt_dBm_vec/10)
Pt_vec=Pt_mW_vec*10**(-3)

#Channel gain map (CGM) based on ray tracing (RT) data using Wireless Insite Software
#CGM is able to predict the channel gains between each tx-rx pairs
#CGM: with the locations of the tx and rx as the input, and the predicted channel gain at each sub-band n as the output, denoted as h_kj[n] 
#Input dimension: (4,) (tx-rx coordinates)
#Output dimension: (N,) (number of sub-bands), the predicted channel gain over the N sub-bands
class ChannelGainMap_RT_Based:
    def __init__(self,x_min=-350.0, x_max=0, y_min=-350.0, y_max=0, gainMin_dB=GAIN_MIN_dB,gainMax_dB=GAIN_MAX_dB):
        self.INPUT_DIM=4 #tx-rx locations
        self.NUM_CHs=len(fc_MHz) #number of sub-bands
        self.X_MIN=x_min
        self.X_MAX=x_max
        self.Y_MIN=y_min
        self.Y_MAX=y_max
        self.CG_MIN=gainMin_dB
        self.CG_MAX=gainMax_dB
        
        self.database_input=np.zeros(shape=(0,self.INPUT_DIM), dtype=np.float64)
        #the database storing the normalized coordinates of the tx-rx pairs, (M,4) 
        #the first two elements correspond to normalized tx locations, and the last two for normalized rx locations
        #normalization w.r.t. X_MAX and Y_MAX, so the input range is [0, 1]
        self.database_label=np.zeros(shape=(0,self.NUM_CHs),dtype=np.float64)
        #the database storing the normalized channel gains  with respect to interval [CG_MIN, CG_MAX]
        #to range [0, 1]
        self.model = self.create_model()      
 
    
    def create_model(self):
        inp=Input(shape=(self.INPUT_DIM,))
        outp=Dense(64,activation='relu')(inp)
        outp=Dense(128,activation='relu')(outp)
        outp=Dense(256,activation='relu')(outp)
        outp=Dense(512,activation='relu')(outp)
        outp=Dense(256,activation='relu')(outp)
        outp=Dense(128,activation='relu')(outp)
        outp=Dense(64,activation='relu')(outp)
        outp=Dense(self.NUM_CHs,activation='linear')(outp)
                    
        model=Model(inp,outp)        
        model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_absolute_error', 'mean_squared_error'])                
        model.summary()
        return model
      
    
    #new_input: (M,INPUT_DIM), locations given in meters
    #new_label: (M,NUM_CHs), channel gain given in dB
    def add_new_data(self,new_input,new_label):
        if new_input.shape[1]!=self.INPUT_DIM:
            raise Exception('Invalid data input dimension')
        if new_label.shape[1]!=self.NUM_CHs:
            raise Exception('Invalid data label dimension')
        if new_input.shape[0]!=new_label.shape[0]:
            raise Exception('Input data and label dimensions do not match')
            
        new_input=self.normalize_locs(new_input)
        new_label=self.normalize_ch_gain(new_label)
               
        self.database_input=np.concatenate((self.database_input,new_input),axis=0)
        self.database_label=np.concatenate((self.database_label,new_label),axis=0)        

        
    def shuffle_data(self):
         input_data=self.database_input
         label_data=self.database_label
                
         input_data,label_data=shuffle(input_data,label_data)#randomly shuffle the data
         self.database_input=input_data
         self.database_label=label_data
        
    
    def normalize_locs(self,tx_rx_locs):#normalize the location coordinate to the range between 0 and 1 
        tx_rx_locs[tx_rx_locs[:,0]<self.X_MIN,0]=self.X_MIN
        tx_rx_locs[tx_rx_locs[:,2]<self.X_MIN,2]=self.X_MIN
        tx_rx_locs[tx_rx_locs[:,0]>self.X_MAX,0]=self.X_MAX
        tx_rx_locs[tx_rx_locs[:,2]>self.X_MAX,2]=self.X_MAX
        
        tx_rx_locs[tx_rx_locs[:,1]<self.Y_MIN,1]=self.Y_MIN
        tx_rx_locs[tx_rx_locs[:,3]<self.Y_MIN,3]=self.Y_MIN
        tx_rx_locs[tx_rx_locs[:,1]>self.Y_MAX,1]=self.Y_MAX
        tx_rx_locs[tx_rx_locs[:,3]>self.Y_MAX,3]=self.Y_MAX
        
        x_length=self.X_MAX-self.X_MIN
        y_length=self.Y_MAX-self.Y_MIN
                
        xy_ranges=np.array([x_length,y_length,x_length,y_length])
        xy_min=np.array([self.X_MIN,self.Y_MIN,self.X_MIN,self.Y_MIN])
        tx_rx_loc_normalized=(tx_rx_locs-xy_min)/xy_ranges
                
        return tx_rx_loc_normalized
            
    
    def normalize_ch_gain(self,ch_gain):#normalize the channel gain to the range betwee 0 and 1 
        ch_gain[ch_gain>self.CG_MAX]=self.CG_MAX
        ch_gain[ch_gain<self.CG_MIN]=self.CG_MIN
        ch_gain_normalized=(ch_gain-self.CG_MIN)/(self.CG_MAX-self.CG_MIN)  
        return ch_gain_normalized
                   
    def predict_channel_gain(self,tx_rx_locs):
        tx_rx_locs_normalized=self.normalize_locs(tx_rx_locs)
        ch_gain_normalized=self.model.predict(tx_rx_locs_normalized)
        ch_gain_dB=self.CG_MIN+ch_gain_normalized*(self.CG_MAX-self.CG_MIN)
        ch_gain_dB[ch_gain_dB>self.CG_MAX]=self.CG_MAX
        ch_gain_dB[ch_gain_dB<self.CG_MIN]=self.CG_MIN

        return ch_gain_dB
        
    def train_map(self,verbose_on=1):        
        self.shuffle_data()
        
        data_size=self.database_input.shape[0]
        
        train_data=self.database_input[:int(data_size*0.9),:]
        train_label=self.database_label[:int(data_size*0.9),:]
             
        test_data=self.database_input[int(data_size*0.9):,:]
        test_label=self.database_label[int(data_size*0.9):,:]
             
        history=self.model.fit(train_data,train_label,epochs=30,validation_split=0.1,verbose=verbose_on)
        
        self.model.save("D2D_model")
        
                    
        history_dict = history.history
        history_dict.keys()
                                                                
        mse = history_dict['mean_squared_error']
        val_mse = history_dict['val_mean_squared_error']
        mae = history_dict['mean_absolute_error']
        val_mae=history_dict['val_mean_absolute_error']
             
        epochs = range(1, len(mse) + 1)
        
        fig=plt.figure()           
        plt.plot(epochs, mse, 'bo', label='Training MSE')
        plt.plot(epochs, val_mse, 'r', label='Validation MSE')
        plt.title('Training and validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()        
        plt.show()
        
        file_dir='results/'
        
        fig_name=file_dir+'DNN_train_MSE'
        fig.savefig(fig_name+'.eps')
        fig.savefig(fig_name+'.pdf')
        fig.savefig(fig_name+'.jpg')

                           
        fig=plt.figure()        
        plt.plot(epochs, mae, 'bo', label='Training MAE')
        plt.plot(epochs, val_mae, 'r', label='Validation MAE')
        plt.title('Training and validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()       
        plt.show()
        
        fig_name=file_dir+'DNN_train_MAE'
        fig.savefig(fig_name+'.eps')
        fig.savefig(fig_name+'.pdf')
        fig.savefig(fig_name+'.jpg')
        
             
        result=self.model.evaluate(test_data,test_label)
        print(result)
                
    

name_strs=['_15.r012.p2m','_22.r020.p2m','_34.r033.p2m']
NUM_LOC_SETS=len(name_strs) #total number of location sets

def read_ray_tracing_data():
    RT_tx_locs_all=np.array([],dtype=np.float64).reshape(0,2)#(M,2), 2D transmitter locations from the ray tracing results
    RT_rx_locs_all=np.array([],dtype=np.float64).reshape(0,2)#(M,2), corresponding receiver locations. M is the number of tx-rx pairs
    RT_ch_gain_dB_all=np.array([],dtype=np.float64).reshape(0,len(fc_MHz))#(M,N), corresponding channel gain in dB, N is the number of sub-bands

    for loc_set in range(NUM_LOC_SETS):   
        RT_tx_locs_cur=np.array([],dtype=np.float64).reshape(0,2)#(M,2), 2D transmitter locations from the ray tracing results
        RT_rx_locs_cur=np.array([],dtype=np.float64).reshape(0,2)#(M,2), corresponding receiver locations. M is the number of tx-rx pairs
        RT_ch_gain_dB_cur=np.array([],dtype=np.float64).reshape(0,len(fc_MHz))#(M,N), corresponding channel gain in dB, N is the number of sub-bands
                                                               
        file_dir='RayTracingD2D/location-set-'+str(loc_set+1)
        print('Currently reading loc set:')
        print(loc_set)
        
        tx_locs=np.loadtxt(fname=file_dir+'/Tx-Locations.txrx')
        tx_locs=tx_locs[:,:2]#only keep the x-y coordinate since height is fixed to 1.5 meter
        
        file_name=file_dir+'/'+str(int(fc_MHz[0]))+'MHz/ad_hoc_mobile_network.pg.t001'+name_strs[loc_set]               
        ray_tracing_data=np.loadtxt(fname=file_name)
        rx_locs=ray_tracing_data[:,1:3]
        
        for tx_idx in range(tx_locs.shape[0]):     
            if tx_idx+1<10:
                tx_idx_str='00'+str(tx_idx+1)
            elif tx_idx+1<100:
                tx_idx_str='0'+str(tx_idx+1)
            else:
                tx_idx_str=str(tx_idx+1)
                    
            RT_rx_locs_cur=np.concatenate((RT_rx_locs_cur,rx_locs),axis=0)
                        
            tx_locs_aug=np.repeat(tx_locs[tx_idx,:].reshape(1,2),rx_locs.shape[0],axis=0)       
            
            RT_tx_locs_cur=np.concatenate((RT_tx_locs_cur,tx_locs_aug),axis=0)
            
            ch_gain_cur=np.array([]).reshape(rx_locs.shape[0],0)
            for freq in fc_MHz:                
                file_name=file_dir+'/'+str(int(freq))+'MHz/ad_hoc_mobile_network.pg.t'                  
                file_name=file_name+tx_idx_str+name_strs[loc_set]    
                ray_tracing_data=np.loadtxt(fname=file_name)    
                ch_gain_dB=ray_tracing_data[:,5].reshape(-1,1)
                ch_gain_cur=np.concatenate((ch_gain_cur,ch_gain_dB),axis=1)
                
           
            RT_ch_gain_dB_cur=np.concatenate((RT_ch_gain_dB_cur,ch_gain_cur),axis=0)
            
            
        RT_tx_locs_all=np.concatenate((RT_tx_locs_all,RT_tx_locs_cur),axis=0)
        RT_rx_locs_all=np.concatenate((RT_rx_locs_all,RT_rx_locs_cur),axis=0)
        RT_ch_gain_dB_all=np.concatenate((RT_ch_gain_dB_all,RT_ch_gain_dB_cur),axis=0)
        
    
    RT_ch_gain_dB_all[RT_ch_gain_dB_all<GAIN_MIN_dB]=GAIN_MIN_dB 
    RT_ch_gain_dB_all[RT_ch_gain_dB_all>GAIN_MAX_dB]=GAIN_MAX_dB
    
    RT_ch_gain_dB_cur[RT_ch_gain_dB_cur<GAIN_MIN_dB]=GAIN_MIN_dB
    RT_ch_gain_dB_cur[RT_ch_gain_dB_cur>GAIN_MAX_dB]=GAIN_MAX_dB
        
    return RT_tx_locs_all,RT_rx_locs_all,RT_ch_gain_dB_all,RT_tx_locs_cur,RT_rx_locs_cur,RT_ch_gain_dB_cur#also return the result for the last set of sampling points for D2D pair selection

           
RT_tx_locs_all,RT_rx_locs_all,RT_ch_gain_dB_all,RT_tx_locs_last_set,RT_rx_locs_last_set,RT_ch_gain_dB_last_set=read_ray_tracing_data()

np.savez('D2D_RT_data',RT_tx_locs_all,RT_rx_locs_all,RT_ch_gain_dB_all,RT_tx_locs_last_set,RT_rx_locs_last_set,RT_ch_gain_dB_last_set)

npzfile = np.load('D2D_RT_data.npz')
RT_tx_locs_all=npzfile['arr_0']
RT_rx_locs_all=npzfile['arr_1']
RT_ch_gain_dB_all=npzfile['arr_2']
RT_tx_locs_last_set=npzfile['arr_3']
RT_rx_locs_last_set=npzfile['arr_4']
RT_ch_gain_dB_last_set=npzfile['arr_5']


#K: number of D2D pairs
#D2D_ch_gain_min: minimum channel gain for D2D link
#Choose the D2D pairs from the last location set
def choose_D2D_pairs(K=30, D2D_ch_gain_min=-80.0): #choose the D2D pairs from the ray tracing results 
    ch_gain_band1=RT_ch_gain_dB_last_set[:,0]
    indices=(ch_gain_band1>=D2D_ch_gain_min)
    tx_locs_valid=RT_tx_locs_last_set[indices,:]
    rx_locs_valid=RT_rx_locs_last_set[indices,:]
    
    tx_locs_valid,rx_locs_valid=shuffle(tx_locs_valid,rx_locs_valid)#randomly shuffle the data

    tx_loc_D2D=np.zeros(shape=(0,2))
    rx_loc_D2D=np.zeros(shape=(0,2))
    
    for k in range(K):
        tx_cur=tx_locs_valid[0,:].reshape(-1,2)
        rx_cur=rx_locs_valid[0,:].reshape(-1,2)
        
        tx_loc_D2D=np.concatenate((tx_loc_D2D,tx_cur),axis=0)
        rx_loc_D2D=np.concatenate((rx_loc_D2D,rx_cur),axis=0)
        
        #remove all links associated with the selected tx or rx, since each of them
        #can only be involved in one D2D link
        check1=tx_locs_valid[:,0]==tx_cur[0,0]
        check2=tx_locs_valid[:,1]==tx_cur[0,1]
        
        tx_locs_valid=tx_locs_valid[~(check1 & check2),:]
        rx_locs_valid=rx_locs_valid[~(check1 & check2),:]
        
        check3=rx_locs_valid[:,0]==rx_cur[0,0]
        check4=rx_locs_valid[:,1]==rx_cur[0,1]
        
        tx_locs_valid=tx_locs_valid[~(check3 & check4),:]
        rx_locs_valid=rx_locs_valid[~(check3 & check4),:]
           
    return tx_loc_D2D,rx_loc_D2D
                

tx_loc_D2D,rx_loc_D2D=choose_D2D_pairs(K=30,D2D_ch_gain_min=-80.0) #(K,2)

num_D2D_pairs=tx_loc_D2D.shape[0]


def plot_D2D_pairs(tx_locs,rx_locs,fig,ax):
    ax.plot(tx_locs[:,0],tx_locs[:,1],'b^',markersize=5)
    ax.plot(rx_locs[:,0],rx_locs[:,1],'ro',markersize=5)
    for i in range(tx_locs.shape[0]):
        plt.plot([tx_locs[i,0],rx_locs[i,0]],[tx_locs[i,1],rx_locs[i,1]])
        ax.annotate(str(i+1),xy=((tx_locs[i,0]+rx_locs[i,0])/2,(tx_locs[i,1]+rx_locs[i,1])/2))
               
    
fig,ax=plt.subplots()        
plot_D2D_pairs(tx_loc_D2D,rx_loc_D2D,fig,ax)
plt.xlabel('x (meter)',fontsize=14)
plt.ylabel('y (meter)',fontsize=14)
plt.show()
 

#Exclude the links associated with the selected D2D tx-rxs to get the data for map training
#For K D2D pairs, a total of K**2 links are excluded
def get_new_data_exclude_D2D_pairs(tx_locs,rx_locs):
    K=rx_locs.shape[0]#number of D2D pairs
    
    RT_tx_locs_new=RT_tx_locs_all    
    RT_rx_locs_new=RT_rx_locs_all
    RT_ch_gain_dB_new=RT_ch_gain_dB_all
    
    RT_tx_locs_test0=np.array([]).reshape(0,RT_tx_locs_new.shape[1])    
    RT_rx_locs_test0=np.array([]).reshape(0,RT_rx_locs_new.shape[1])
    RT_ch_gain_dB_test0=np.array([]).reshape(0,RT_ch_gain_dB_new.shape[1]) 
               
    for tx in range(K):   
        tx_cur=tx_locs[tx,:]  
        for rx in range(K):                    
            tx_check=np.isclose(RT_tx_locs_new,tx_cur, rtol=1e-04, atol=1e-04)                    
            rx_cur=rx_locs[rx,:]
            rx_check=np.isclose(RT_rx_locs_new, rx_cur, rtol=1e-04, atol=1e-04)                
            mask_tx_rx_idx=rx_check[:,0] & rx_check[:,1] & tx_check[:,0] & tx_check[:,1]      
   
            RT_tx_locs_test0=np.concatenate((RT_tx_locs_test0,RT_tx_locs_new[mask_tx_rx_idx,:]),axis=0)
            RT_rx_locs_test0=np.concatenate((RT_rx_locs_test0,RT_rx_locs_new[mask_tx_rx_idx,:]),axis=0)
            RT_ch_gain_dB_test0=np.concatenate((RT_ch_gain_dB_test0,RT_ch_gain_dB_new[mask_tx_rx_idx,:]),axis=0)
            
            RT_tx_locs_new=RT_tx_locs_new[~mask_tx_rx_idx,:]
            RT_rx_locs_new=RT_rx_locs_new[~mask_tx_rx_idx,:]
            RT_ch_gain_dB_new=RT_ch_gain_dB_new[~mask_tx_rx_idx,:] 
            
              
    return RT_tx_locs_new,RT_rx_locs_new,RT_ch_gain_dB_new,RT_tx_locs_test0,RT_rx_locs_test0,RT_ch_gain_dB_test0  


RT_tx_locs_new,RT_rx_locs_new,RT_ch_gain_dB_new,RT_tx_locs_test0,RT_rx_locs_test0,RT_ch_gain_dB_test0=get_new_data_exclude_D2D_pairs(tx_loc_D2D,rx_loc_D2D)


def split_test_train_data(tx_locs,testSamplePerTx=1000):    
    RT_tx_locs_test=np.array([]).reshape(0,RT_tx_locs_new.shape[1])    
    RT_rx_locs_test=np.array([]).reshape(0,RT_rx_locs_new.shape[1])
    RT_ch_gain_dB_test=np.array([]).reshape(0,RT_ch_gain_dB_new.shape[1]) 
    
    RT_tx_locs_train=np.array([]).reshape(0,RT_tx_locs_new.shape[1])    
    RT_rx_locs_train=np.array([]).reshape(0,RT_rx_locs_new.shape[1])
    RT_ch_gain_dB_train=np.array([]).reshape(0,RT_ch_gain_dB_new.shape[1]) 
       
    RT_tx_locs_left=RT_tx_locs_new
    RT_rx_locs_left=RT_rx_locs_new
    RT_ch_gain_dB_left=RT_ch_gain_dB_new
    
    K=tx_locs.shape[0]
    for tx in range(K):
        tx_cur=tx_locs[tx,:]        
        tx_check=np.isclose(RT_tx_locs_left,tx_cur, rtol=1e-04, atol=1e-04)  
        
        mask_tx=tx_check[:,0] & tx_check[:,1]
        #all data associated with current TX
        RT_tx_locs_cur=RT_tx_locs_left[mask_tx,:]
        RT_rx_locs_cur=RT_rx_locs_left[mask_tx,:]
        RT_ch_gain_dB_cur=RT_ch_gain_dB_left[mask_tx,:]
        
        
        RT_tx_locs_left=RT_tx_locs_left[~mask_tx,:]
        RT_rx_locs_left=RT_rx_locs_left[~mask_tx,:]
        RT_ch_gain_dB_left=RT_ch_gain_dB_left[~mask_tx,:]
         
        #split the data associated with the current tx
        cur_size=RT_tx_locs_cur.shape[0]
        test_indices=np.random.choice(cur_size,testSamplePerTx,replace=False)
        
        RT_tx_locs_test=np.concatenate((RT_tx_locs_test,RT_tx_locs_cur[test_indices,:]),axis=0)
        RT_rx_locs_test=np.concatenate((RT_rx_locs_test,RT_rx_locs_cur[test_indices,:]),axis=0)
        RT_ch_gain_dB_test=np.concatenate((RT_ch_gain_dB_test,RT_ch_gain_dB_cur[test_indices,:]),axis=0)
        
        all_rxs=np.arange(cur_size)
        train_indices=np.delete(all_rxs,test_indices)
        
        RT_tx_locs_train=np.concatenate((RT_tx_locs_train,RT_tx_locs_cur[train_indices,:]),axis=0)
        RT_rx_locs_train=np.concatenate((RT_rx_locs_train,RT_rx_locs_cur[train_indices,:]),axis=0)
        RT_ch_gain_dB_train=np.concatenate((RT_ch_gain_dB_train,RT_ch_gain_dB_cur[train_indices,:]),axis=0)
        
        
    RT_tx_locs_train=np.concatenate((RT_tx_locs_train,RT_tx_locs_left),axis=0)
    RT_rx_locs_train=np.concatenate((RT_rx_locs_train,RT_rx_locs_left),axis=0)
    RT_ch_gain_dB_train=np.concatenate((RT_ch_gain_dB_train,RT_ch_gain_dB_left),axis=0)
    
    return RT_tx_locs_train,RT_rx_locs_train,RT_ch_gain_dB_train,RT_tx_locs_test,RT_rx_locs_test,RT_ch_gain_dB_test 


RT_tx_locs_train,RT_rx_locs_train,RT_ch_gain_dB_train,RT_tx_locs_test,RT_rx_locs_test,RT_ch_gain_dB_test=split_test_train_data(tx_loc_D2D,testSamplePerTx=1000)   

RT_tx_locs_test=np.concatenate((RT_tx_locs_test,RT_tx_locs_test0),axis=0)
RT_rx_locs_test=np.concatenate((RT_rx_locs_test,RT_rx_locs_test0),axis=0)
RT_ch_gain_dB_test=np.concatenate((RT_ch_gain_dB_test,RT_ch_gain_dB_test0),axis=0)    
               
        
#Curve fitting for the path loss model to find the channel gain intercept, negative of path loss exponent, and the pre-log term for the frequency
def PL_model_curve_fitting(tx_locs_train, rx_locs_train, ch_gain_dB_train):
    distances=LA.norm(tx_locs_train-rx_locs_train,axis=1).reshape(-1,1)
    N=ch_gain_dB_train.shape[1]
    data_size=ch_gain_dB_train.shape[0]
    
    print(ch_gain_dB_train.shape)
    
    ch_gain_dB_train_flatterned=ch_gain_dB_train.flatten().reshape(-1,1)
    distances_aug=np.repeat(distances,N,axis=0)
    
    fc_MHz_aug=fc_MHz.reshape(-1,1)
    fc_MHz_aug=np.tile(fc_MHz_aug,(data_size,1))
    
    A=np.concatenate((np.ones(fc_MHz_aug.shape),10*np.log10(distances_aug),10*np.log10(fc_MHz_aug)),axis=1)
    
    PL_model_par_vec=np.matmul(np.linalg.pinv(A),ch_gain_dB_train_flatterned)#the least square curve fitting
    
    return PL_model_par_vec
    
PL_model_par_vec=PL_model_curve_fitting(RT_tx_locs_train,RT_rx_locs_train,RT_ch_gain_dB_train)

print(PL_model_par_vec)


#predict the channel based on curve-fitted path loss model
#input:
#PL_model_par_vec: (3,1), the parameter for the channel gain intercept, negative of path loss exponent, and the pre-log term for the frequency
def pred_channel_curve_fitted_PL_model(PL_model_par_vec,tx_locs,rx_locs):
    K=rx_locs.shape[0]
    rx_locs_aug=np.tile(rx_locs,(K,1))#(K**2,2)
    tx_locs_aug=np.repeat(tx_locs,K,axis=0) #(K**2,2)
    
    ch_gain_dB=pred_channel_curve_fitted_PL_model_given_locs(PL_model_par_vec,tx_locs_aug,rx_locs_aug)       
    return ch_gain_dB,tx_locs_aug,rx_locs_aug
    
#PL_model_par_vec: (3,1), the parameter for the channel gain intercept, negative of path loss exponent, and the pre-log term for the frequency
def pred_channel_curve_fitted_PL_model_given_locs(PL_model_par_vec,tx_locs_aug,rx_locs_aug):   
    distances=LA.norm(rx_locs_aug-tx_locs_aug,axis=1).reshape(-1,1)
    #(K**2,1), the link distances between each of the K txs with the K rxs, including both the desired links and interfering links 
    N=len(fc_MHz)    
    distances_aug=np.repeat(distances,N,axis=0)
        
    fc_MHz_aug=fc_MHz.reshape(-1,1)
    fc_MHz_aug=np.tile(fc_MHz_aug,(distances.shape[0],1))
    
    A=np.concatenate((np.ones(fc_MHz_aug.shape),10*np.log10(distances_aug),10*np.log10(fc_MHz_aug)),axis=1)    
    ch_gain_dB=np.matmul(A,PL_model_par_vec)
    ch_gain_dB=ch_gain_dB.reshape(N,-1,order='F')#(N,K**2)
    ch_gain_dB=ch_gain_dB.T #(K**2,N)
    
    return ch_gain_dB


def get_true_RT_D2D_link_channels(tx_locs,rx_locs):
    K=rx_locs.shape[0]#number of D2D pairs    
    H_true_RT_dB=np.zeros((K,K,len(fc_MHz)),dtype=np.float64)
    
    for rx in range(K):
        rx_cur=rx_locs[rx,:]
        rx_check=np.isclose(RT_rx_locs_all,rx_cur, rtol=1e-04, atol=1e-04)
        
        for tx in range(K):
            tx_cur=tx_locs[tx,:]        
            tx_check=np.isclose(RT_tx_locs_all,tx_cur, rtol=1e-04, atol=1e-04)           
            mask_tx_rx_idx=rx_check[:,0] & rx_check[:,1] & tx_check[:,0] & tx_check[:,1]      
            
            H_true_RT_dB[rx,tx,:]=RT_ch_gain_dB_all[mask_tx_rx_idx,:]            
    return H_true_RT_dB
            

#Get the true D2D channel gains based on the Ray tracing result           
H_true_RT_dB=get_true_RT_D2D_link_channels(tx_loc_D2D,rx_loc_D2D)   
H_true_RT=10**(H_true_RT_dB/10)
               
#Get the D2D channel gains based on the curve-fitted PL model
channel_PL_dB_fitted, tx_locs_aug,rx_locs_aug=pred_channel_curve_fitted_PL_model(PL_model_par_vec,tx_loc_D2D,rx_loc_D2D)
H_PL_fitted_dB=np.reshape(channel_PL_dB_fitted,(num_D2D_pairs,num_D2D_pairs,-1),order='F') #(K,K,N)
#H_PL_fitted_dB=np.reshape(H_PL_fitted_dB,(num_D2D_pairs,num_D2D_pairs,-1),order='F') #(K,K,N)
H_PL_fitted=10**(H_PL_fitted_dB/10)


cg_map_RT=ChannelGainMap_RT_Based()   
cg_map_RT.add_new_data(np.concatenate((RT_tx_locs_train,RT_rx_locs_train),axis=1),RT_ch_gain_dB_train)

#data augmentation based on tx-rx reciprocity
#swap the tx-rx, the channel should be the same
#cg_map_RT.add_new_data(np.concatenate((RT_rx_locs_train,RT_tx_locs_train),axis=1),RT_ch_gain_dB_train)


#print('All data shape:')
#print(RT_tx_locs_all.shape,RT_rx_locs_all.shape,RT_ch_gain_dB_all.shape)
print('Train data shape:')
print(cg_map_RT.database_input.shape)      

cg_map_RT.train_map()



#predict the channels using the trained channel gain map
tx_rx_locs=np.concatenate((tx_locs_aug,rx_locs_aug),axis=1)
channel_map_pred_dB=cg_map_RT.predict_channel_gain(tx_rx_locs)#(K**2,N)
H_map_pred_dB=np.reshape(channel_map_pred_dB,(num_D2D_pairs,num_D2D_pairs,-1),order='F') #(K,K,N)
H_map_pred=10**(H_map_pred_dB/10)



#=========Get the true communication rate=======
#H_true: (K,K,N), the true CSI information for calculating the communication rate
#channel_assignment: the given channel assignment: nested list of length N, containing the user indices that are assigned to each channel n   
#Pt: transmit power
#noise_power
def get_true_rate(H_true,channel_assignment,Pt,noise_power):    
    N=H_true.shape[2] #Number of sub-channels       
    rate_each_channel=np.zeros(N) #The sum rate for each sub-channel
    
    for n in range(N):
        K_n=channel_assignment[n]# users assigned to sub-channel n
        R_n=0
                  
        for j in K_n:#Calculate the rate of each existing user j using sub-channel n, if user k is also assigned to sub-channel n
            SINR_j=Pt*H_true[j,j,n]/(Pt*np.sum(H_true[j,K_n,n])-Pt*H_true[j,j,n]+noise_power)            
            R_n=R_n+np.log2(1+SINR_j)
        
        rate_each_channel[n]=R_n
   
    sum_rate=np.sum(rate_each_channel)
    
    return sum_rate


#=========The greedy algorithm for channel assignment=======
#H_known: (K,K,N), the assumed CSI information used for channel assignment  
#Pt: transmit power
#noise_power: noise power
#order_list: the order list of all the K users
def greedy_channel_assignment_given_order(H_known,Pt,noise_power,order_list):
    N=H_known.shape[2] #Number of sub-channels
 
    user_each_channel = [[] for i in range(N)] #A list of length N, containing the user indices that are assigned to each channel n   
    rate_each_channel=np.zeros(N) #Current sum rate for each sub-channel
    
    for k in order_list:#assign sub-channel to each user sequentially
        cand_sum_rate=np.zeros(N) #candidate sum rate if sub-channel n is assigned to user k
        
        T_n_vec=np.zeros(N)#The resulting new sum-rate for each sub-channel n if it is assigned to user k 
        for n in range(N):
            K_n=user_each_channel[n]#current users assigned to sub-channel n
                
            SINR_k=Pt*H_known[k,k,n]/(Pt*np.sum(H_known[k,K_n,n])+noise_power)
            R_kn=np.log2(1+SINR_k) #rate of user k, if it is assigned to sub-channel n            
            T_n=R_kn
            
            for j in K_n:#Calculate the rate of each existing user j using sub-channel n, if user k is also assigned to sub-channel n
                SINR_j=Pt*H_known[j,j,n]/(Pt*np.sum(H_known[j,K_n,n])-Pt*H_known[j,j,n]+Pt*H_known[j,k,n]+noise_power)
                T_n=T_n+np.log2(1+SINR_j)
                
            T_n_vec[n]=T_n #store the resulting rate at sub-channel n, if it is also assigned to user k
            
            cand_sum_rate[n]=T_n+np.sum(rate_each_channel[:n])+np.sum(rate_each_channel[n+1:])
            
        
        n_star=np.argmax(cand_sum_rate)#select the sub-channel resulting the maximum sum-rate
        rate_each_channel[n_star]=T_n_vec[n_star]#update the rate for the selected sub-channel
        
        temp=user_each_channel[n_star]
        temp.append(k)
        user_each_channel[n_star]=temp
        
    sum_rate=np.sum(rate_each_channel)
            
    return user_each_channel,sum_rate

def generate_random_permutations(K,TOTAL_TRIALS):
    random_orders_all=np.array([],dtype=np.int).reshape(K,0)#(M,2), 2D transmitter locations from the ray tracing results
    for i in range(TOTAL_TRIALS):
        random_order=np.random.permutation(K).reshape(-1,1)
        random_orders_all=np.concatenate((random_orders_all,random_order),axis=1)
        
    return random_orders_all
        

def greedy_channel_assignment(H_known,Pt,noise_power,random_orders_all):
    sum_rate=0
    for i in range(random_orders_all.shape[1]):
        random_order=random_orders_all[:,i]
        user_each_channel_cur,sum_rate_cur=greedy_channel_assignment_given_order(H_known,Pt,noise_power,random_order)
        
        if sum_rate_cur>sum_rate:
            sum_rate=sum_rate_cur
            user_each_channel=user_each_channel_cur
            
    return user_each_channel
            

def channel_assign_all_schemes(TOTAL_TRIALS=200):
    sum_rate_CSI_knowledge,sum_rate_map_knowledge,sum_rate_PL_fitted_knowledge=[],[],[]
          
    K=H_true_RT.shape[0] #number of D2D pairs
    
    random_orders_all=generate_random_permutations(K,TOTAL_TRIALS)    

    for i in range(len(Pt_vec)):
        Pt=Pt_vec[i]         
        
        #channel assignment with perfect CSI
        channel_assign_CSI_knowledge=greedy_channel_assignment(H_true_RT,Pt,noise_power,random_orders_all)
        sum_rate_CSI_knowledge_cur=get_true_rate(H_true_RT,channel_assign_CSI_knowledge,Pt,noise_power)
        sum_rate_CSI_knowledge.append(sum_rate_CSI_knowledge_cur)
                       
        #channel assignment with curve-fitted PL knowledge
        channel_assign_PL_fitted_knowledge=greedy_channel_assignment(H_PL_fitted,Pt,noise_power,random_orders_all)
        sum_rate_PL_knowledge_fitted_cur=get_true_rate(H_true_RT,channel_assign_PL_fitted_knowledge,Pt,noise_power)
        sum_rate_PL_fitted_knowledge.append(sum_rate_PL_knowledge_fitted_cur)
         
        #channel assignment with map-predicted channel
        channel_assign_map_knowledge=greedy_channel_assignment(H_map_pred,Pt,noise_power,random_orders_all)
        sum_rate_map_knowledge_cur=get_true_rate(H_true_RT,channel_assign_map_knowledge,Pt,noise_power)
        sum_rate_map_knowledge.append(sum_rate_map_knowledge_cur)
               
       
    print('Sum rate for channel assignment with perfect CSI, map knowledge, and fitted path loss:')
    print(sum_rate_CSI_knowledge)
    print(sum_rate_map_knowledge)
    print(sum_rate_PL_fitted_knowledge)
    
    
    fig=plt.figure()
    plt.xlabel('Transmit power (dBm)',fontsize=14)
    plt.ylabel('Sum rate (bps/Hz)',fontsize=14)
    plt.plot(Pt_dBm_vec,sum_rate_CSI_knowledge,'k-^',linewidth=2,label='Perfect CSI-based')
    plt.plot(Pt_dBm_vec,sum_rate_map_knowledge,'b-o',linewidth=2,label='CGM-based')
    plt.plot(Pt_dBm_vec,sum_rate_PL_fitted_knowledge,'r-+',linewidth=2,label='Fitted PL model-based')
    plt.legend(loc="upper left",prop={'size': 14})
    plt.grid(b=True, which='major', color='#999999', linestyle='-')
    plt.show()

    fig_name='results/D2Dsumrate'
    fig.savefig(fig_name+'.eps')
    fig.savefig(fig_name+'.pdf')
    fig.savefig(fig_name+'.jpg')
    #plt.ylim(-6000,0)


channel_assign_all_schemes(TOTAL_TRIALS=200)


def compare_channel_pred_acc():
    normalized_MAE_map_based=np.abs(H_map_pred_dB[:]-H_true_RT_dB[:])/np.abs(H_true_RT_dB[:])
    normalized_MAE_map_based=normalized_MAE_map_based.reshape(-1,1)
    
    normalized_MAE_fitted_PL_based=np.abs(H_PL_fitted[:]-H_true_RT_dB[:])/np.abs(H_true_RT_dB[:])
    normalized_MAE_fitted_PL_based=normalized_MAE_fitted_PL_based.reshape(-1,1)
    

    plt.subplot(211)
    plt.hist(normalized_MAE_fitted_PL_based,bins=50)
#    plt.xlim([0,1.0])
    plt.title('Fitted PL model based')
    plt.xlabel('normalized MAE')
    plt.subplot(212)
    plt.hist(normalized_MAE_map_based,bins=50)
#    plt.xlim([0,1.0])
    plt.title('Map based')
    plt.xlabel('normalized MAE')
    
    
    print('Mean MAE: map knowledge and fitted PL:')
    print(np.mean(normalized_MAE_map_based),np.mean(normalized_MAE_fitted_PL_based)) 


compare_channel_pred_acc()

RX_SAMPLES_PER_TX=1848 #number of location samples per transmitter 
   
    

def plot_channel_gain_maps(tx_locs_D2D,rx_locs_D2D):   
    for k in range(5):
        tx_cur=tx_locs_D2D[k,:]
        
        tx_check=np.isclose(RT_tx_locs_test,tx_cur, rtol=1e-04, atol=1e-04)
               
        mask_tx_rx_idx= tx_check[:,0] & tx_check[:,1]      
                        
        rx_locs=RT_rx_locs_test[mask_tx_rx_idx,:]#all rxs corresponding to the current tx
        ch_gain_dB_all_bands=RT_ch_gain_dB_test[mask_tx_rx_idx,:]
               
        ch_gain_dB=ch_gain_dB_all_bands[:,0]
        
        fig,ax=plt.subplots()

        xgrid=np.linspace(rx_locs[:,0].min(),rx_locs[:,0].max(),100)
        ygrid=np.linspace(rx_locs[:,1].min(),rx_locs[:,1].max(),100)
        xgrid,ygrid=np.meshgrid(xgrid,ygrid)
        zgrid=griddata(rx_locs,ch_gain_dB,(xgrid,ygrid))
        
        v = np.linspace(GAIN_MIN_dB-10,GAIN_MAX_dB, 14,endpoint=True)
        tick=v.flatten()
        plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
        cbar=plt.colorbar(ticks=tick)
        
        
        plt.plot(tx_cur[0], tx_cur[1], '^',markersize=15)
        ax.annotate('tx'+str(k+1), xy=tx_cur,fontsize=14)
           
        plt.xlabel('x (meter)',fontsize=14)
        plt.ylabel('y (meter)',fontsize=14)
        plt.title('true',fontsize=14)
        
        cbar.set_label('Channel gain [dB]',labelpad=20, rotation=270,fontsize=14)
        
        file_dir='results/'
        
        fig_name=file_dir+'true_channel_gain_map_tx'+str(k+1)
        fig.savefig(fig_name+'.eps')
        fig.savefig(fig_name+'.pdf')
        fig.savefig(fig_name+'.jpg')

        plt.show()
        
        
        tx_cur_aug=np.repeat(tx_cur.reshape(1,2),rx_locs.shape[0],axis=0)
        
        ch_gain_dB_all_bands_pred=cg_map_RT.predict_channel_gain(np.concatenate((tx_cur_aug,rx_locs),axis=1))
        
        ch_gain_dB_pred=ch_gain_dB_all_bands_pred[:,0]
        

        fig,ax=plt.subplots()
        
        
        xgrid=np.linspace(rx_locs[:,0].min(),rx_locs[:,0].max(),100)
        ygrid=np.linspace(rx_locs[:,1].min(),rx_locs[:,1].max(),100)
        xgrid,ygrid=np.meshgrid(xgrid,ygrid)
        zgrid=griddata(rx_locs,ch_gain_dB_pred,(xgrid,ygrid))
        
        v = np.linspace(GAIN_MIN_dB-10,GAIN_MAX_dB, 14,endpoint=True)
        tick=v.flatten()
        plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
        cbar=plt.colorbar(ticks=tick)
        
        
        plt.plot(tx_cur[0], tx_cur[1], '^',markersize=15)
        ax.annotate('tx'+str(k+1), xy=tx_cur,fontsize=14)
                
        
        plt.xlabel('x (meter)',fontsize=14)
        plt.ylabel('y (meter)',fontsize=14)
        plt.title('DNN-based',fontsize=14)
        
        cbar.set_label('Channel gain [dB]',labelpad=20, rotation=270,fontsize=14)
        fig_name=file_dir+'pred_channel_gain_map_tx'+str(k+1)
        fig.savefig(fig_name+'.eps')
        fig.savefig(fig_name+'.pdf')
        fig.savefig(fig_name+'.jpg')

        plt.show()
        
        
        
        channel_PL_fitted_dB_all_bands=pred_channel_curve_fitted_PL_model_given_locs(PL_model_par_vec,tx_cur_aug,rx_locs)
        channel_PL_fitted_dB=channel_PL_fitted_dB_all_bands[:,0]
        fig,ax=plt.subplots()
        
        xgrid=np.linspace(rx_locs[:,0].min(),rx_locs[:,0].max(),100)
        ygrid=np.linspace(rx_locs[:,1].min(),rx_locs[:,1].max(),100)
        xgrid,ygrid=np.meshgrid(xgrid,ygrid)
        zgrid=griddata(rx_locs,channel_PL_fitted_dB,(xgrid,ygrid))
                
        #v = np.linspace(ch_gain_dB.min(), ch_gain_dB.max(), 10)
        v = np.linspace(GAIN_MIN_dB-10,GAIN_MAX_dB, 14,endpoint=True)
        tick=v.flatten()
        plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
        cbar=plt.colorbar(ticks=tick)
        
        
        plt.plot(tx_cur[0], tx_cur[1], '^',markersize=15)
        ax.annotate('tx'+str(k+1), xy=tx_cur,fontsize=14)

                       
        plt.xlabel('x (meter)',fontsize=14)
        plt.ylabel('y (meter)',fontsize=14)
        plt.title('Fitted PL model-based',fontsize=14)
        
        cbar.set_label('Channel gain [dB]',labelpad=20, rotation=270,fontsize=14)
        fig_name=file_dir+'channel_gain_map_PL_fitted_model_tx'+str(k+1)
        fig.savefig(fig_name+'.eps')
        fig.savefig(fig_name+'.pdf')
        fig.savefig(fig_name+'.jpg')

        plt.show()
               
        
plot_channel_gain_maps(tx_loc_D2D,rx_loc_D2D) 

#np.savez('D2DChannelAssignment_Main') 







