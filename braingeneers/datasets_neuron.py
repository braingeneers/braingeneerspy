import numpy as np
import musclebeachtools as mbt
import glob
import sys
import os
import re
import functools

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output


s3 = 'aws --endpoint https://s3.nautilus.optiputer.net s3'
res_path = 'mgk_results/'
s3_path = 'braingeneers/ephys/'
data_path = 'data/'



def get_well_dict(exp,data_path):
    '''
    Load well dict from s3 experiment, used for pathing of exp and well to data location
    
    
    Parameters:
    -----------
    exp:  str
          uuid of experiment directory followed by '/'
    data_path:  str
          location to path where local data is stored.
          
    Returns:
    --------
    well_dict: dict of dicts
          A dict which indexes based on experiments. The values are also dicts, which index based on well, with the 
          value being the path of the experiment-well'''
    
    # Sort
    loc = f'{data_path}{exp}root/data/{exp}results/*'
    chs = sorted(glob.glob(loc))
    

    # Remove rasters 
    chs = [ch for ch in chs if 'rasters' not in ch]

    well_dict = {}
    seen_ch = []

    for ch in chs:

        well_grp = ch.split('/')[-1]
        well, grp = well_grp.split('chgroup')

        #Maps group to full name
        temp = {well+grp:ch}

        if well not in seen_ch:
            seen_ch.append(well)
            well_dict[well] = {}

        well_dict[well][int(grp)] = ch


    return well_dict
    
    
    
############# Experiment Loading Functions ###################    
def load_experiment(exp:str):
    '''Load experiment from s3 to local, return well dict.
    
    Parameters:
    -----------
    exp:    str
            name of experiment followed by '/' (ex. test1/)
    
    Global vars:
    ------------
    data_path:  str
            path where data will be downloaded
    
    Returns:
    --------
    well_dict: dict
               A dict which indexes based on experiments. The values are also dicts,
               which index based on well, with the value being the
               path of the experiment-well'''
    
    
    #Check if experiment already has been downloaded
    if os.path.isdir(data_path + exp):
        well_dict = get_well_dict(exp, data_path = data_path)
        
        #TODO: Check if this is going to cause an issue
        #!{s3} cp s3://{s3_path + exp + 'dataset/'} {data_path}{exp}
        print('Selected! Experiment already exists locally:',exp)
        return well_dict
    
    print('Loading experiment: ' + exp[:-1])
    
    # Download files from exp
    !{s3} cp s3://{s3_path + exp + res_path} data/{exp}. --recursive --exclude="*" --include="*.zip"
        
    #Unzip those files
    !unzip -qn 'data/{exp}*.zip' -d data/{exp[:-1]}
    !rm data/{exp}*.zip
    
    
    well_dict = get_well_dict(exp, data_path = 'data/')
    
    print('Finished loading:',exp)
    return (well_dict)



def load_experiment_b(b,well_dict,selected_exp):
  '''
  Button interfacing with 

  '''
    global well_dict
    exp = select_exp.value
    well_dict = load_experiment(exp)
    
    return
    
def generate_load_experiment_b(well_dict,selected_exp):
  '''
  Generates button for loading experiment
  '''
  load_btn = widgets.Button(description="Load")
  load_btn.on_click(functools.partial(load_experiment_b,well_dict,selected_exp))
  display(load_btn)
  

def load_well(well,well_dict,exp):
    '''Loads corresponding wells neurons and ratings
    
    Arguments:
    well -- location of well (ex. 'A1')
    exp -- name of experiment followed by '/' (ex. test1/)
    
    Global vars:
    data_path -- path where data will be downloaded
    
    '''
    
    neurons = None
    
    #Sort by actual number
    well_data = {k: v for k,v in sorted(well_dict[well].items(),key=lambda x: x[0])}
    
    #Load and append each group to the data list, accumulating potential spikes
    for group in well_data.values():
        nf = glob.glob(group + '/spikeintf/outputs/neurons*')
        n_temp = np.load(nf[0],allow_pickle=True)
        n_prb = open(glob.glob(group+'/spikeintf/inputs/*probefile.prb')[0])
        mbt.load_spike_amplitudes(n_temp, group+'/spikeintf/outputs/amplitudes0.npy')
        
        lines = n_prb.readlines()
        real_chans = []
        s = lines[5]
        n = s.split()
        
        for chan in range(1,len(n)):
            result = re.search('c_(.*)\'', n[chan])
            real_chans.append(int(result.group(1)))

        for i in range(len(n_temp)):
            chan = n_temp[i].peak_channel
            n_temp[i].peak_channel = real_chans[chan]

        if type(neurons) != np.ndarray:
            neurons = n_temp
        else:
            neurons = np.append(neurons,n_temp)

        n_prb.close()

    print('Well {} selected, {} potential neurons!'.format(well,len(neurons)),end='')
    
    #Load ratings and assign
    ratings_path = f'{data_path}{exp}dataset/'
    ratings_path = glob.glob(ratings_path + well + '*.npy')
    
    if len(ratings_path) > 0:
        ratings = np.load(ratings_path[0])
        print('{} ratings loaded.'.format(len(ratings)))
        
    else:
        print('Ratings NOT loaded.')
    
    
    return (neurons,ratings)



def load_well_b(b):
    '''Loads all channel groups from the specified well selected in the drop down menu'''
    global neurons
    global fs
    global ratings
    
    neurons = None
    
    #Load from dropdowns
    well = select_well.value
    exp = select_exp.value
    
    neurons,ratings = load_well(well,well_dict,exp)
    
    fs = neurons[0].fs
    return


def load_well_raw(well,well_dict,exp):
    '''Loads corresponding wells raw electrode data
    
    Arguments:
    well -- location of well (ex. 'A1')
    exp -- name of experiment followed by '/' (ex. test1/)
    
    Global vars:
    data_path -- path where data will be downloaded
    
    '''
    
    neurons = []
    #Sort by actual number
    well_data = {k: v for k,v in sorted(well_dict[well].items(),key=lambda x: x[0])}
    
    #Load and append each group to the data list, accumulating raw
    for group in well_data.values():
        nf = glob.glob(group + '/spikeintf/outputs/neurons*')
        
        print(nf)
        n_temp = np.load(nf[0],allow_pickle=True)
        n_prb = open(glob.glob(group+'/spikeintf/inputs/*probefile.prb')[0])
        mbt.load_spike_amplitudes(n_temp, group+'/spikeintf/outputs/amplitudes0.npy')
        
        lines = n_prb.readlines()
        real_chans = []
        s = lines[5]
        n = s.split()
        
        for chan in range(1,len(n)):
            result = re.search('c_(.*)\'', n[chan])
            real_chans.append(int(result.group(1)))

        for i in range(len(n_temp)):
            chan = n_temp[i].peak_channel
            n_temp[i].peak_channel = real_chans[chan]

        if type(neurons) != np.ndarray:
            neurons = n_temp
        else:
            neurons = np.append(neurons,n_temp)

        n_prb.close()    

    
    
    
    
    
    
    
    

    
def get_well_data(well_dict,stim_period=None):
    '''
    Loads data from specific well in well_dict.
    
    Parameters:
    -----------
    well_dict: dict
              Dictionary of the well groups returned by get_well()
              This looks like get_data_well(well_dict['A1'])
    stim_period: int
              How the data can be split over a 3rd dim, cutting the time into chunks of
              {stim_period} seconds.
    
    Returns:
    --------
    Tuple- 
    Data: np.array
          n,k,t array of neurons, stims, stim_period
    fs:   int
          Sampling freq
    neu:  list
          List of neuron objects
    '''
    arrs = []
    neu = []
    n = neuron.Neuron('temp')
    fs = 0
    for pref in well_dict.values():
    
        #Load file
        nf = glob.glob(pref + '/spikeintf/outputs/neurons*')
        data = np.load(nf[0],allow_pickle=True)
        fs = data[0].fs

        #Make dense
        spike_list = [data[i].spike_time for i in range(len(data))]
        arrs.append(n.load_spike_times(spike_list,max_neurons=100))
        
        af = glob.glob(pref + '/spikeintf/outputs/amplitudes0*')
        data = mbt.load_spike_amplitudes(data, af[0])

        neu = neu + list(data)
        
        #Shorten data to shortest of them all
    data = shorten_all_fs(arrs,fs)
    
    
    if stim_period is not None:
        #Make data fit under multiple of stim_periods
        data = shorten_fs(data,stim_period)
        data = data.reshape((data.shape[0],data.shape[1]//stim_period,stim_period))
    return (data,fs,neu)
    
    
    
    
def shorten_all_fs(nd,fs):
    min_time = min([i.shape[1] for i in nd])
    cut_amount = int(min_time%fs)
    cut_ind = min_time - cut_amount
    
    return np.vstack([i[:,:cut_ind] for i in nd])

def shorten_fs(nd,fs):
    cut_amount = int(nd.shape[1]%fs)
    cut_ind = nd.shape[1] - cut_amount
    
    return nd[:,:cut_ind]