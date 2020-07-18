# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:33:50 2020

@author: Yong Zeng
mmWaveBF_Main
Environmental-aware millimeter wave beamforming in urbarn environment
Three schemes are considered:
    1. Perfect CSI-based mmWave beam selection: the BS has the perfect knowledge of the MISO channel
    2. Channel path map (CPM)-based mmWave beam selection: the BS is able to predict the path knowledge (delay, path gain, AoD) of the L strongest paths, and reconstruct the MISO channel
    3. Location-based mmWave beam selection: the BS knows the location of the UE and select the beam based on the transmit array response of the location
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.patches import Ellipse


EPS=10**-6 #A small value to ensure numerical stability

BS_LOC=np.array([[173.0, 112.0, 73.3355]])


RT_TX_POWER_dBm=30 #required to calculate the path gain based on the path power

Pt_dBm=40
Pt_mW=10**(Pt_dBm/10)
Pt=Pt_mW*10**(-3)


noise_power_density_dBm_Hz=-174.0
BW=10.0*10**6
noise_power_mW=10**(noise_power_density_dBm_Hz/10)*BW
noise_power=noise_power_mW*10**(-3)

Nz_vec=np.array([10,10,10,10,10,10,10,10,10])
Ny_vec=np.array([5,10,20,30,40,50,80,100,160])


LOC_ERR_MEAN_VEC=np.array([0,1.0,5.0,10.0])#Standard deviation of the localization error
N_vec=Nz_vec*Ny_vec


NUM_PATHS_TO_LEARN=3
INFORMATION_PER_PATH=4
POWER_SENSITIVITY=-300 #in dBm. A sufficiently small value


RX_SETS=['r051','r052']

PATH_FILE_DIR='RayTracingMmWaveBF/GridRxs.paths.t001_46.'
RX_LOC_FILE_DIR='RayTracingMmWaveBF/GridRxs.pg.t001_46.'

def read_ray_tracing_data():
    rx_locs=read_rx_locs()
       
    path_num=np.array([],dtype=np.int16)
    path_knowl_all=np.array([],dtype=np.float64).reshape(0,4)#(M,4), M is the total paths of all RXs    
    path_knowl_L_strongest=np.array([],dtype=np.float64).reshape(0,NUM_PATHS_TO_LEARN*INFORMATION_PER_PATH)#(K,12), K is the number of RXs
    
    for rx_name_str in RX_SETS:
        path_file_name=PATH_FILE_DIR+rx_name_str+'.p2m'
        
        with open(path_file_name,'r') as f:
            all_lines=f.readlines()
        
        Rx_num=int(all_lines[21].strip('\n'))
    
        path_num_curset=np.array([],dtype=np.int16)
        path_knowl_all_curset=np.array([],dtype=np.float64).reshape(0,4)#(M,4), M is the total paths of all RXs
        path_knowl_L_strongest_curset=np.array([],dtype=np.float64).reshape(0,NUM_PATHS_TO_LEARN*INFORMATION_PER_PATH)#(K,12), K is the number of RXs
        
    
        line_idx=22 #start from the first path
        for Rx in range(Rx_num):
            data=all_lines[line_idx].split()    
            total_path=int(data[1])
            path_num_curset=np.append(path_num_curset,total_path)
           
            if total_path==0:
                line_idx=line_idx+1
            else:
                line_idx=line_idx+2;#main data for the first path           
    
                path_knowl_cur=np.array([],dtype=np.float64).reshape(0,4)
                for path in range(total_path):
                    data=all_lines[line_idx].split()
                    interactions=int(data[1])              
                    power_phase_theta_phi=np.array([[float(data[2]),float(data[3]),float(data[7]),float(data[8])]])                
                    path_knowl_cur=np.concatenate((path_knowl_cur,power_phase_theta_phi),axis=0)
                                   
                    line_idx=line_idx+(4+interactions); # consider the line specify interaction and interaction locations
                           
                
                path_knowl_all_curset=np.concatenate((path_knowl_all_curset,path_knowl_cur),axis=0)
            
            
            #Extract the L strongest paths
            strong_paths_knowl_cur=np.zeros((1,NUM_PATHS_TO_LEARN*INFORMATION_PER_PATH))
            min_temp=np.minimum(total_path,NUM_PATHS_TO_LEARN)
            
            indices=np.arange(min_temp)*INFORMATION_PER_PATH
            
            strong_paths_indices=np.argsort(-path_knowl_cur[:,0])[:min_temp]
            strong_paths_knowl_cur[0,indices]=path_knowl_cur[strong_paths_indices,0]
            strong_paths_knowl_cur[0,indices+1]=path_knowl_cur[strong_paths_indices,1]
            strong_paths_knowl_cur[0,indices+2]=path_knowl_cur[strong_paths_indices,2]
            strong_paths_knowl_cur[0,indices+3]=path_knowl_cur[strong_paths_indices,3]
            
            if total_path<NUM_PATHS_TO_LEARN:#pad with some special values if the number of paths is insufficient
                indices=np.arange(min_temp,NUM_PATHS_TO_LEARN)*INFORMATION_PER_PATH
                strong_paths_knowl_cur[0,indices]=POWER_SENSITIVITY
                strong_paths_knowl_cur[0,indices+1]=-400
                strong_paths_knowl_cur[0,indices+2]=-400
                strong_paths_knowl_cur[0,indices+3]=-400
            
          
            path_knowl_L_strongest_curset=np.concatenate((path_knowl_L_strongest_curset,strong_paths_knowl_cur),axis=0) 
            
            
            
        path_num=np.append(path_num,path_num_curset)
        path_knowl_all=np.concatenate((path_knowl_all,path_knowl_all_curset),axis=0)
        path_knowl_L_strongest=np.concatenate((path_knowl_L_strongest,path_knowl_L_strongest_curset),axis=0)
        
   
    
    path_knowl_all[:,0]=path_knowl_all[:,0]-RT_TX_POWER_dBm#Get the path gain based on received power and transmitted power 
    indices=np.arange(NUM_PATHS_TO_LEARN)*INFORMATION_PER_PATH
    path_knowl_L_strongest[:,indices]=path_knowl_L_strongest[:,indices]-RT_TX_POWER_dBm

    return rx_locs,path_num,path_knowl_L_strongest,path_knowl_all

def exclude_no_path_rxs(rx_locs,path_num,path_knowl_L_strongest):
    mask=path_num==0
    rx_locs_new=rx_locs[~mask,:]
    path_num_new=path_num[~mask]
    path_knowl_L_strongest_new=path_knowl_L_strongest[~mask,:]
    
    return rx_locs_new, path_num_new, path_knowl_L_strongest_new
    
    
def read_rx_locs():
    rx_locs=np.array([],dtype=np.float64).reshape(0,3)
    for rx_name_str in RX_SETS:
        fname=RX_LOC_FILE_DIR+rx_name_str+'.p2m'
        rx_locs_cur=np.loadtxt(fname)
        rx_locs_cur=rx_locs_cur[:,[1,2,3]]
        rx_locs=np.concatenate((rx_locs,rx_locs_cur),axis=0)
        
    return rx_locs

#Reconstruct the MISO channel given the path information
#UPA is placed on the y-z plane, with N=Ny x Nz elements
#path_num: a list storing the number of paths for each rx
#path_knowl: (total_paths,4), storing the path power, phase, zenith AoD, and azimuth AoD
def reconstruct_channel(path_num,path_knowl,Ny,Nz):
    Rx_num=len(path_num)
    Channel=np.zeros((Ny*Nz,Rx_num),dtype=complex)
    for rx in range(Rx_num):
        start_idx=int(np.sum(path_num[:rx]))
        total_path=int(path_num[rx])
        channel_cur=0       
        for path in range(total_path):
            power_dBm=path_knowl[start_idx+path,0]
            power_mW=10**(power_dBm/10)
            power_W=power_mW*10**(-3)
            
            phase_deg=path_knowl[start_idx+path,1]
            phase_rad=np.deg2rad(phase_deg)
            
            zenith_deg=path_knowl[start_idx+path,2]
            azimuth_deg=path_knowl[start_idx+path,3]
            
            a_response=get_array_response(zenith_deg,azimuth_deg,Ny,Nz)
            
            channel_cur=channel_cur+np.sqrt(Ny*Nz)*np.sqrt(power_W)*np.exp(1j*phase_rad)*a_response
            
        Channel[:,rx]=channel_cur
        
    return Channel
                        
#Get the array response for the given AoD (theta,phi) in degree    
def get_array_response(theta,phi,Ny,Nz):   
    dy=0.5
    dz=0.5#Ajacent antenna elements seperation in wave-length
    ny_vec=np.arange(Ny)
    nz_vec=np.arange(Nz)
    
    theta_rad=np.deg2rad(theta)
    phi_rad=np.deg2rad(phi)
    
    ay=(1/np.sqrt(Ny))*np.exp(1j*2*np.pi*dy*np.sin(theta_rad)*np.sin(phi_rad)*ny_vec)
    
    az=(1/np.sqrt(Nz))*np.exp(1j*2*np.pi*dz*np.cos(theta_rad)*nz_vec)
    
    a=np.kron(ay,az)
    
    return a
    

#rx_locs,path_num,path_knowl_L_strongest,path_knowl_all=read_ray_tracing_data() 
#np.savez('MmWaveBF_RT_data',rx_locs,path_num,path_knowl_L_strongest,path_knowl_all)


npzfile = np.load('MmWaveBF_RT_data.npz')
rx_locs=npzfile['arr_0']
path_num=npzfile['arr_1']
path_knowl_L_strongest=npzfile['arr_2']
path_knowl_all=npzfile['arr_3']


def plot_strongest_path_AoD_map():
    rx_locs_plot_3D,path_num_plot,path_knowl_L_strongest_plot,path_knowl_plot,rx_locs_train,path_num_train,path_knowl_L_strongest_train=select_test_rxs(3000,rx_locs, path_num, path_knowl_L_strongest)
    rx_locs_plot=rx_locs_plot_3D[:,:2]
    
    #Plot the azimuth angle of departure of the strongest path
    true_labels=path_knowl_L_strongest_plot[:,3]
    sin_true_labels=np.sin(np.deg2rad(true_labels))
    no_path_indices=true_labels<-180.0
    sin_true_labels[no_path_indices]=1.5    
    xgrid=np.linspace(rx_locs_plot[:,0].min(),rx_locs_plot[:,0].max(),200)
    ygrid=np.linspace(rx_locs_plot[:,1].min(),rx_locs_plot[:,1].max(),200)
    xgrid,ygrid=np.meshgrid(xgrid,ygrid)    
    zgrid=griddata(rx_locs_plot,sin_true_labels,(xgrid,ygrid))   
    v = np.linspace(-1.0,1, 21,endpoint=True)
    v=np.append(v,2)
    tick=v.flatten()
    fig=plt.figure()
    plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
    cbar=plt.colorbar(ticks=tick)    
    plt.xlabel('x (m)',fontsize=14)
    plt.ylabel('y (m)',fontsize=14)
    plt.title('azimuth AoD map: true')   
    cbar.set_label('sin($\phi$)',labelpad=20, rotation=270,fontsize=14)    
    file_dir='results/'        
    fig_name=file_dir+'azimuthAoD_True'
    fig.savefig(fig_name+'.eps')
    fig.savefig(fig_name+'.pdf')
    fig.savefig(fig_name+'.jpg')

    
    #KNN-based map
    fig=plt.figure()
    pred_output_KNN=pred_KNN(rx_locs_train[:,:2],path_knowl_L_strongest_train,rx_locs_plot,K=3) 
    pred_phi=pred_output_KNN[:,3]
    sin_pred_phi=np.sin(np.deg2rad(pred_phi))
    no_path_indices=pred_output_KNN[:,3]<-180.0
    sin_pred_phi[no_path_indices]=1.5 
    xgrid=np.linspace(rx_locs_plot[:,0].min(),rx_locs_plot[:,0].max(),200)
    ygrid=np.linspace(rx_locs_plot[:,1].min(),rx_locs_plot[:,1].max(),200)
    xgrid,ygrid=np.meshgrid(xgrid,ygrid)    
    zgrid=griddata(rx_locs_plot,sin_pred_phi,(xgrid,ygrid))
    tick=v.flatten()
    plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
    cbar=plt.colorbar(ticks=tick)
    plt.xlabel('x (m)',fontsize=14)
    plt.ylabel('y (m)',fontsize=14)
    plt.title('azimuth AoD map: IDW-KNN-based')  
    cbar.set_label('sin($\phi$)',labelpad=21, rotation=270,fontsize=14)   
    file_dir='results/'        
    fig_name=file_dir+'azimuthAoD_KNNBased'
    fig.savefig(fig_name+'.eps')
    fig.savefig(fig_name+'.pdf')
    fig.savefig(fig_name+'.jpg')
    
    #Location-based
    fig=plt.figure()
    loc_based_AoD=get_loc_based_AoD(BS_LOC,rx_locs_plot_3D)
    cos_theta_loc_based=np.cos(np.deg2rad(loc_based_AoD[:,0]))
    sin_phi_loc_based=np.sin(np.deg2rad(loc_based_AoD[:,1]))
    xgrid=np.linspace(rx_locs_plot[:,0].min(),rx_locs_plot[:,0].max(),200)
    ygrid=np.linspace(rx_locs_plot[:,1].min(),rx_locs_plot[:,1].max(),200)
    xgrid,ygrid=np.meshgrid(xgrid,ygrid)    
    zgrid=griddata(rx_locs_plot,sin_phi_loc_based,(xgrid,ygrid))
    tick=v.flatten()
    plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
    cbar=plt.colorbar(ticks=tick)
    plt.xlabel('x (m)',fontsize=14)
    plt.ylabel('y (m)',fontsize=14)
    plt.title('azimuth AoD map: loc.-based')  
    cbar.set_label('sin($\phi$)',labelpad=21, rotation=270,fontsize=14)   
    file_dir='results/'        
    fig_name=file_dir+'azimuthAoD_locBased'
    fig.savefig(fig_name+'.eps')
    fig.savefig(fig_name+'.pdf')
    fig.savefig(fig_name+'.jpg')
              
    #Plot the zenith angular of departure of the strongest path
    true_labels=path_knowl_L_strongest_plot[:,2]
    cos_true_labels=np.cos(np.deg2rad(true_labels))
    no_path_indices=true_labels<-180.0
    cos_true_labels[no_path_indices]=1.5   
    xgrid=np.linspace(rx_locs_plot[:,0].min(),rx_locs_plot[:,0].max(),200)
    ygrid=np.linspace(rx_locs_plot[:,1].min(),rx_locs_plot[:,1].max(),200)
    xgrid,ygrid=np.meshgrid(xgrid,ygrid)    
    zgrid=griddata(rx_locs_plot,cos_true_labels,(xgrid,ygrid))    
    v = np.linspace(-1.0,1, 21,endpoint=True)
    v=np.append(v,2)
    tick=v.flatten()
    fig=plt.figure()
    plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
    cbar=plt.colorbar(ticks=tick)   
    plt.xlabel('x (m)',fontsize=14)
    plt.ylabel('y (m)',fontsize=14)
    plt.title('zenith AoD map: true')   
    cbar.set_label('cos($\\theta$)',labelpad=20, rotation=270,fontsize=14)   
    file_dir='results/'       
    fig_name=file_dir+'zenithAoD_True'
    fig.savefig(fig_name+'.eps')
    fig.savefig(fig_name+'.pdf')
    fig.savefig(fig_name+'.jpg')

    

    fig=plt.figure()
    pred_output_KNN=pred_KNN(rx_locs_train[:,:2],path_knowl_L_strongest_train,rx_locs_plot,K=3) 
    pred_theta=pred_output_KNN[:,2]
    cos_pred_theta=np.cos(np.deg2rad(pred_theta))
    no_path_indices=pred_output_KNN[:,2]<-180.0
    cos_pred_theta[no_path_indices]=1.5   
    xgrid=np.linspace(rx_locs_plot[:,0].min(),rx_locs_plot[:,0].max(),200)
    ygrid=np.linspace(rx_locs_plot[:,1].min(),rx_locs_plot[:,1].max(),200)
    xgrid,ygrid=np.meshgrid(xgrid,ygrid)    
    zgrid=griddata(rx_locs_plot,cos_pred_theta,(xgrid,ygrid))
    tick=v.flatten()
    plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
    cbar=plt.colorbar(ticks=tick)
    plt.xlabel('x (m)',fontsize=14)
    plt.ylabel('y (m)',fontsize=14)
    plt.title('zenith AoD map: KNN-based')    
    cbar.set_label('cos($\\theta$)',labelpad=21, rotation=270,fontsize=14)  
    file_dir='results/'       
    fig_name=file_dir+'zenithAoD_KNNBased'
    fig.savefig(fig_name+'.eps')
    fig.savefig(fig_name+'.pdf')
    fig.savefig(fig_name+'.jpg')
    
    
    
    fig=plt.figure()    
    xgrid=np.linspace(rx_locs_plot[:,0].min(),rx_locs_plot[:,0].max(),200)
    ygrid=np.linspace(rx_locs_plot[:,1].min(),rx_locs_plot[:,1].max(),200)
    xgrid,ygrid=np.meshgrid(xgrid,ygrid)    
    zgrid=griddata(rx_locs_plot,cos_theta_loc_based,(xgrid,ygrid))
    tick=v.flatten()
    plt.contourf(xgrid,ygrid,zgrid, cmap=cm.jet,levels=tick)
    cbar=plt.colorbar(ticks=tick)
    plt.xlabel('x (m)',fontsize=14)
    plt.ylabel('y (m)',fontsize=14)
    plt.title('zenith AoD map: loc.-based')    
    cbar.set_label('cos($\\theta$)',labelpad=21, rotation=270,fontsize=14)  
    file_dir='results/'       
    fig_name=file_dir+'zenithAoD_LocBased'
    fig.savefig(fig_name+'.eps')
    fig.savefig(fig_name+'.pdf')
    fig.savefig(fig_name+'.jpg')
    
   

#rx_locs_new,path_num_new,path_knowl_L_strongest_new,path_knowl_all=read_ray_tracing_data() 
rx_locs_new, path_num_new, path_knowl_L_strongest_new=exclude_no_path_rxs(rx_locs,path_num,path_knowl_L_strongest)
   
def select_test_rxs(NUM_TESTS,rx_locs_input,path_num_input,path_knowl_L_strongest_input):#randomly select test rx locations
    K=rx_locs_input.shape[0]#total number of rx locations
    selected_indices=np.random.choice(K,NUM_TESTS,replace=False)
    
    rx_locs_test=rx_locs_input[selected_indices,:]
    path_num_test=path_num_input[selected_indices]
    path_knowl_L_strongest_test=path_knowl_L_strongest_input[selected_indices,:]
    
    all_rxs=np.arange(K)
    train_indices=np.delete(all_rxs,selected_indices)
    
    rx_locs_train=rx_locs_input[train_indices,:]
    path_num_train=path_num_input[train_indices]
    path_knowl_L_strongest_train=path_knowl_L_strongest_input[train_indices,:]
    
    path_knowl_test=np.array([]).reshape(0,INFORMATION_PER_PATH)
    for rx in selected_indices:
        start=np.sum(path_num_input[:rx])
        end=start+path_num_input[rx]
        path_knowl_test=np.concatenate((path_knowl_test,path_knowl_all[start:end,:]),axis=0)
          
    return rx_locs_test,path_num_test,path_knowl_L_strongest_test,path_knowl_test,rx_locs_train,path_num_train,path_knowl_L_strongest_train
    

rx_locs_test,path_num_test,path_knowl_L_strongest_test,path_knowl_test,rx_locs_train,path_num_train,path_knowl_L_strongest_train=select_test_rxs(1000,rx_locs_new, path_num_new, path_knowl_L_strongest_new)

print('Number of train locations:'+str(rx_locs_train.shape[0]))
print('Number of test locations:'+str(rx_locs_test.shape[0]))


    
#prediction based on K nearest neighor algorithm
#the labelled data is stored in rx_locs_train and path_knowl_L_strongest_train
#locs_database: (M,2)
#path_knowl_database: (M,NUM_PATHS_TO_LEARN*INFORMATION_PER_PATH)
#query_locs: (N,2)
def pred_KNN(locs_database,path_knowl_database, query_locs,K=3):
    N=query_locs.shape[0]
    pred_path_knowl=np.zeros((N,path_knowl_database.shape[1]))
    for n in range(N):
        loc_cur=query_locs[n,:]
        neighbor_indices,neighbor_distances=get_K_nearest_neighbors(locs_database,loc_cur,K)
        neighbor_distances=np.maximum(neighbor_distances,EPS)
        distances_inverse=1./neighbor_distances
        weights=distances_inverse/np.sum(distances_inverse)
        neighbor_labels=path_knowl_database[neighbor_indices,:]
        pred_path_knowl[n,:]=np.matmul(weights,neighbor_labels)
        
    return pred_path_knowl
        

def get_K_nearest_neighbors(locs_database,loc_cur,K):
    distances=LA.norm(locs_database-loc_cur,axis=1)
    neighbor_indices=np.argpartition(distances,K)[:K]
    neighbor_distances=distances[neighbor_indices]
    
    return neighbor_indices,neighbor_distances
    

#Given the BS and UE locations, get the location-based AoD
#BS_loc: (1,3)
#UE_locs:(M,3)    
def get_loc_based_AoD(BS_loc,UE_locs):
    directions=UE_locs-BS_loc
    distances=LA.norm(directions,axis=1)
    distances=np.maximum(distances,EPS)
    thetas=np.rad2deg(np.arccos(directions[:,2]/distances))
    phis=np.rad2deg(np.arctan2(directions[:,1],directions[:,0]))
    
    loc_based_AoD=np.concatenate((thetas.reshape(-1,1),phis.reshape(-1,1)),axis=1)
    
    return loc_based_AoD
    
        
#theta_range: (2,), the minimum and maximum values of zenith angle theta
#phi_range:(2,) the minimum and maximum values of azimuth angle phi           
def generate_BF_codebook(theta_range,Ny,Nz):
    resol_z=1/(2*Nz)
    resol_y=1/(2*Ny)
    
    cos_theta_range=np.cos(np.deg2rad(theta_range))
    
    sin_phi_range=np.array([-1.0,1.0])

    LB=np.maximum(cos_theta_range[1],-1)
    UB=np.minimum(cos_theta_range[0]+resol_z,1)
    
    sample_cos_theta=np.arange(LB,UB,resol_z)
    sample_sin_phi=np.arange(np.maximum(sin_phi_range[0],-1),np.minimum(sin_phi_range[1]+resol_y,1),resol_y)
    
    sample_theta=np.arccos(sample_cos_theta)
    sample_theta=np.rad2deg(sample_theta)
    
    sample_phi=np.arcsin(sample_sin_phi)
    sample_phi=np.rad2deg(sample_phi)

       
    sample_theta_aug=np.tile(sample_theta,len(sample_phi))
    
    sample_phi_aug=np.repeat(sample_phi,len(sample_theta))
    
    ANGLE_GRID=np.concatenate((sample_theta_aug.reshape(-1,1),sample_phi_aug.reshape(-1,1)),axis=1)
    
    CODE_BOOK=np.zeros((len(sample_theta_aug),Ny*Nz),dtype=complex)
        
    
    for idx in range(len(sample_theta_aug)):
        CODE_BOOK[idx,:]=get_array_response(sample_theta_aug[idx],sample_phi_aug[idx],Ny,Nz)
        
    return CODE_BOOK,ANGLE_GRID
    
#CODE_BOOK:(M,N), M is the total number of beams, N is the number of antennas
#H: (N,K) the K channels
def beam_selection(CODE_BOOK,H):
    BF_result=np.matmul(np.conj(CODE_BOOK),H)
    
    gains=np.square(np.absolute(BF_result))

    
    BF_indices=np.argmax(gains,axis=0)
    max_gains=np.amax(gains,axis=0)
    
    return BF_indices, max_gains
    

#=========Get the true communication rate for point-to-point channel=======
#H_true: (N,K), the true channel for the K rx locations, N is the number of transmit antennas
#CODE_BOOK: (M,N), M is the total number of beams, N is the number of transmit antennas 
#BF_indices: (K,), the selected BF indices for each of the K rxs
#Pt: transmit power
#noise_power
def get_true_rate(H_true,CODE_BOOK,BF_indices,Pt,noise_power):    
    K=H_true.shape[1] #Number of rx locations       
    rate_each_locs=np.zeros(K) #The rate for each location
    
    for k in range(K):
        bf=CODE_BOOK[BF_indices[k],:]# selected BF
        gain=np.matmul(np.conjugate(bf),H_true[:,k])
        gain=np.square(np.absolute(gain))
        rate_each_locs[k]=np.log2(1+Pt*gain/noise_power)
           
    return rate_each_locs


#=========Get the true communication rate for multi-user channel=======
#H_true: (N,K_UE), the true channel for the K_UE for which the BF are simultaneously performed, N is the number of transmit antennas
#CODE_BOOK: (M,N), M is the total number of beams, N is the number of transmit antennas 
#BF_indices: (K_UE,), the selected BF indices for each of the K UEs
#Pt: transmit power
#noise_power
def get_true_sum_rate_MU(H_true,CODE_BOOK,BF_indices,Pt,noise_power):    
    K=H_true.shape[1] #Number of rx locations       
    rate_each_UE=np.zeros(K) #The rate for each location
    
    for k in range(K):
        bf=CODE_BOOK[BF_indices[k],:]# selected BF
        gain=np.matmul(np.conjugate(bf),H_true[:,k])
        gain=np.square(np.absolute(gain))
        signal_power=Pt*gain
        
        interference=0
        for j in range(K):
            bfj=CODE_BOOK[BF_indices[j],:]# selected BF
            gain_kj=np.matmul(np.conjugate(bfj),H_true[:,k])
            gain_kj=np.square(np.absolute(gain_kj))
            interference=interference+Pt*gain_kj
            
        interference=interference-signal_power

        rate_each_UE[k]=np.log2(1+signal_power/(interference+noise_power))
           
    return sum(rate_each_UE)


#The localization error is assumed to be haf-normal distributed with standard deviation error_std
#num_real: number of realizations    
def generate_half_normal_loc_error(error_std,num_real):
    rand_disp=error_std*np.random.randn(num_real)
    rand_disp=np.absolute(rand_disp)
    rand_disp=rand_disp.reshape(-1,1)
    
    rand_angle=2*np.pi*np.random.rand(num_real)
    
    direc_vec=np.concatenate((np.cos(rand_angle).reshape(-1,1),np.sin(rand_angle).reshape(-1,1)),axis=1)
    
    loc_error=rand_disp*direc_vec
        
    return loc_error
    

#The localization error is assumed to be Rayleigh distributed with mean_error
#num_real: number of realizations    
def generate_Rayleigh_loc_error(mean_err,num_real):
    scale=mean_err/np.sqrt(np.pi/2)
    rand_disp=np.random.rayleigh(scale,num_real)
#    rand_disp=np.absolute(rand_disp)
    rand_disp=rand_disp.reshape(-1,1)
    
    rand_angle=2*np.pi*np.random.rand(num_real)
    
    direc_vec=np.concatenate((np.cos(rand_angle).reshape(-1,1),np.sin(rand_angle).reshape(-1,1)),axis=1)
    
    loc_error=rand_disp*direc_vec
        
    return loc_error


indices=np.arange(NUM_PATHS_TO_LEARN)*INFORMATION_PER_PATH
TEMP=path_knowl_L_strongest_train[:,indices+2]
TEMP=TEMP[TEMP!=-400.0]
theta_min=np.min(TEMP)
theta_max=np.max(TEMP)
theta_range=np.array([theta_min,theta_max])


def random_choose_UEs(total_UEs,num_sel_UEs,num_rea):
    UE_indices=np.zeros((num_sel_UEs,num_rea),dtype=np.int)
    for rea_idx in range(num_rea):    
        selected_indices=np.random.choice(total_UEs,num_sel_UEs,replace=False)
        UE_indices[:,rea_idx]=selected_indices
        
    return UE_indices
    
       

def single_user_main():
    rate_perfect_CSI=np.zeros((rx_locs_test.shape[0],len(Nz_vec),len(LOC_ERR_MEAN_VEC)))#BF selection based on perfect CSI
    rate_KNN=np.zeros((rx_locs_test.shape[0],len(Nz_vec),len(LOC_ERR_MEAN_VEC)))#BF selection based on channel path knowledge map
    rate_loc_based=np.zeros((rx_locs_test.shape[0],len(Nz_vec),len(LOC_ERR_MEAN_VEC)))#BF selection based on UE location
    
    
    for loc_err_idx in range(len(LOC_ERR_MEAN_VEC)):
        loc_err_cur=LOC_ERR_MEAN_VEC[loc_err_idx]
        loc_errs=generate_Rayleigh_loc_error(loc_err_cur,rx_locs_test.shape[0])
        
        estimated_loc2D=rx_locs_test[:,:2]+loc_errs
        
        estimated_loc3D=np.zeros(rx_locs_test.shape)
        estimated_loc3D[:,:2]=estimated_loc2D
        estimated_loc3D[:,2]=rx_locs_test[:,2]
           
            
        pred_output_KNN=pred_KNN(rx_locs_train[:,:2],path_knowl_L_strongest_train,estimated_loc2D,K=3) 
        #(K,NUM_PATHS_TO_LEARN*INFORMATION_PER_PATH)
          
        K=pred_output_KNN.shape[0]
        pred_path_num=NUM_PATHS_TO_LEARN*np.ones(K,dtype=np.int)
        
        pred_path_knowl_KNN=np.array([]).reshape(0,INFORMATION_PER_PATH)
        for k in range(K):
            for path in range(NUM_PATHS_TO_LEARN):
                temp_cur=pred_output_KNN[k,path*INFORMATION_PER_PATH:(path+1)*INFORMATION_PER_PATH]
                temp_cur=temp_cur.reshape(1,-1)
                pred_path_knowl_KNN=np.concatenate((pred_path_knowl_KNN,temp_cur),axis=0)
    
    
    
        loc_based_AoD=get_loc_based_AoD(BS_LOC,estimated_loc3D)
        
        path_num_loc_based=np.ones(loc_based_AoD.shape[0],dtype=np.int)
        loc_based_knowl=np.zeros((loc_based_AoD.shape[0],4))
        loc_based_knowl[:,0]=-100*np.ones(loc_based_AoD.shape[0],dtype=np.float64)
        loc_based_knowl[:,1]=-100*np.ones(loc_based_AoD.shape[0],dtype=np.float64)
        #The first two columns can be set to arbitrary values, since they do not affect the location-based beam selection
        #due to the one single path
        loc_based_knowl[:,[2,3]]=loc_based_AoD
    
    
        
        for idx in range(len(Nz_vec)):
            Nz=Nz_vec[idx]
            Ny=Ny_vec[idx]
            
            CODE_BOOK,ANGLE_GRID= generate_BF_codebook(theta_range,Ny,Nz)
            
            
            H_true=reconstruct_channel(path_num_test,path_knowl_test,Ny,Nz)
            
            BF_indices_perfect_CSI, gains_perfect_CSI=beam_selection(CODE_BOOK,H_true)
            
            rate_perfect_CSI[:,idx,loc_err_idx]=get_true_rate(H_true,CODE_BOOK,BF_indices_perfect_CSI,Pt,noise_power)
            
            
            H_pred=reconstruct_channel(pred_path_num,pred_path_knowl_KNN,Ny,Nz)
            
            BF_indices_map, gains_map=beam_selection(CODE_BOOK,H_pred)
            
            rate_KNN[:,idx,loc_err_idx]=get_true_rate(H_true,CODE_BOOK,BF_indices_map,Pt,noise_power)
            
            H_loc_based=reconstruct_channel(path_num_loc_based,loc_based_knowl,Ny,Nz)
            
            BF_indices_loc_based,gains_loc_based=beam_selection(CODE_BOOK,H_loc_based)
            rate_loc_based[:,idx,loc_err_idx]=get_true_rate(H_true,CODE_BOOK,BF_indices_loc_based,Pt,noise_power)
        
    
    
    avg_rate_perfect_CSI=np.mean(rate_perfect_CSI,axis=0)
    avg_rate_KNN=np.mean(rate_KNN,axis=0)
    avg_rate_loc_based=np.mean(rate_loc_based,axis=0)
    
    return avg_rate_perfect_CSI, avg_rate_KNN, avg_rate_loc_based
       

avg_rate_perfect_CSI, avg_rate_KNN, avg_rate_loc_based=single_user_main()


for loc_err_idx in range(len(LOC_ERR_MEAN_VEC)):
#        fig=plt.figure()
    plt.xlabel('Number of transmit antennas',fontsize=14)
    plt.ylabel('Average rate (bps/Hz)',fontsize=14)
    plt.plot(N_vec,avg_rate_perfect_CSI[:,loc_err_idx],'-ko',linewidth=2,label='Perfect CSI-based')
    plt.plot(N_vec,avg_rate_KNN[:,loc_err_idx],'-b+',linewidth=2,label='Path map-based')
    plt.plot(N_vec,avg_rate_loc_based[:,loc_err_idx],'-r^',linewidth=2,label='Location-based')
    plt.title('Expected loc. error: '+str(LOC_ERR_MEAN_VEC[loc_err_idx])+'m')
    plt.grid(b=True, which='major', color='#999999', linestyle='-')
    plt.legend(loc="upper left",prop={'size': 14})

    plt.show()
    
    

fig,ax=plt.subplots() 

plot_str=['-b+','-r^','-g*']
ax.plot(N_vec,avg_rate_perfect_CSI[:,0],'-ko',linewidth=2)    
for loc_err_idx in range(len(LOC_ERR_MEAN_VEC)-1):
    ax.plot(N_vec,avg_rate_KNN[:,loc_err_idx],plot_str[loc_err_idx],linewidth=2,label='Loc. err: '+str(int(LOC_ERR_MEAN_VEC[loc_err_idx]))+'m')
    ax.plot(N_vec,avg_rate_loc_based[:,loc_err_idx],plot_str[loc_err_idx],linewidth=2)

#ax.annotate('Perfect CSI',xy=(1000,7.2),fontsize=14)

ax.annotate("Perfect CSI",
            xy=(900, 7.5), xycoords='data',
            xytext=(1200, 7.5), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),fontsize=14
            )

el = Ellipse((400, 5.5), 100, 1.6, edgecolor='k',facecolor='none',alpha=0.5)

ax.add_artist(el)

ax.annotate("Path map-based",
            xy=(450, 5.5), xycoords='data',
            xytext=(750, 5.5), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),fontsize=14
            )
            
el = Ellipse((600, 3.9), 100, 1.6, edgecolor='k',facecolor='none',alpha=0.5)

ax.add_artist(el)
            
ax.annotate("Loc.-based",
            xy=(550, 3.2), xycoords='data',
            xytext=(10, 3.2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),fontsize=14
            )
                
plt.xlabel('Number of transmit antennas',fontsize=14)
plt.ylabel('Average rate (bps/Hz)',fontsize=14)    
plt.grid(b=True, which='major', color='#999999', linestyle='-')
plt.legend(loc="upper left",prop={'size': 14})
plt.show()

file_dir='results/'
        
fig_name=file_dir+'mmWaveAvgRate'
fig.savefig(fig_name+'.eps')
fig.savefig(fig_name+'.pdf')
fig.savefig(fig_name+'.jpg')

plot_strongest_path_AoD_map()   
