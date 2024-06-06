import glob
import logging
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import ipywidgets as widgets
import musclebeachtools as mbt
import numpy as np
from .utils import s3wrangler as wr
from .utils.numpy_s3_memmap import NumpyS3Memmap
from IPython.display import display, clear_output


warnings.filterwarnings("ignore")
logging.disable(sys.maxsize)



s3 = 'aws --endpoint https://s3.nautilus.optiputer.net s3'
res_path = 'mgk_results/'
s3_path = 'braingeneers/ephys/'


class NeuralAid:
    
    
    def __init__(self,data_path='data/'):
        self.data_path = data_path
        self.exp = None
        self.select_exp = None
        self.ratings_dict = get_ratings_dict()
        
        return

    
    def set_data_path(self,new_path):
        
        self.data_path = str(Path(new_path).resolve()) + '/'
        print("Data path changed to:",self.data_path)
        return
   

    def gen_exp_dropdown(self,exps = None,disp=True):
        '''
        Generates dropdown list for experiment selection

        Parameters:
        -----------
        exps: list
                list of experiments returned fro get_exp_list
                if None, generates list
        disp: bool
                if True, displays dropdown immediately
        '''        
        if exps == None:
            exps = get_exp_list()

        self.select_exp = widgets.Dropdown( options=exps, description='Experiment:')
        
        if disp:
            display(self.select_exp)
        return
          
            
    def set_exp(self,exp=None):
        '''
        Sets exp
        If experiment is none, try to pull from dropdown menu
        '''
        if exp == None:
            self.exp = self.select_exp.value
        else:
            if self.select_exp != None:
                self.select_exp.value = exp
            self.exp = exp
        
        return
            
            
    def load_experiment_b(self,b):
        '''
        Button interfacing, set and load experiment

        '''
        self.set_exp()
        print('Loading exp')
        
        self.load_experiment()

        return
    
    def choose_experiment(self):
        self.gen_exp_dropdown()
        self.gen_load_experiment_b()
        return
    
    def choose_well(self):
        self.gen_well_dropdown()
        self.gen_load_well_b()
        
        
    def gen_load_experiment_b(self):
        '''
        Generates button for loading experiment
        '''
        self.load_exp_btn = widgets.Button(description="Load")
        self.load_exp_btn.on_click(self.load_experiment_b)
        
        display(self.load_exp_btn)
        return

    def set_well_dict(self):
        '''
        Load well dict from s3 experiment, used for pathing of exp and well to data location


        Parameters:
        -----------
        self.exp:  str
              uuid of experiment directory followed by '/'
        self.data_path:  str
              location to path where local data is stored.

        Sets:
        --------
        self.well_dict: dict of dicts
              A dict which indexes based on experiments. The values are also dicts, which index based on well, with the 
              value being the path of the experiment-well'''
        
        #Should error out here?
        if self.exp == None:
            self.set_exp()
            
        # Sort
        loc = f'{self.data_path}{self.exp}root/data/{self.exp}results/*'
        chs = sorted(glob.glob(loc))

        # Remove rasters 
        chs = [ch for ch in chs if 'rasters' not in ch]
        self.well_dict = {}
        seen_ch = []

        for ch in chs:

            well_grp = ch.split('/')[-1]
            well, grp = well_grp.split('chgroup')

            # Maps group to full name
            if well not in seen_ch:
                seen_ch.append(well)
                self.well_dict[well] = {}

            self.well_dict[well][int(grp)] = ch
        
        return
    
    
    def get_ratings_dict(self):
        objs = wr.list_objects('s3://braingeneers/ephys/*/dataset/*.npy')

        ratings_dict = {}

        for o in objs:
            #This is dirty
            key = o.split('/')[4] + '/'
            ratings_dict[key] = o
            
        self.ratings_dict = ratings_dict

        return ratings_dict

  
    
    def load_experiment(self,exp=None):
        '''Load experiment from s3 to local, return well dict.

        Parameters:
        -----------
        self.exp:    str
                name of experiment followed by '/' (ex. test1/)
        self.data_path:  str
                path where data will be downloaded

        Sets:
        --------
        self.well_dict: dict
                   A dict which indexes based on experiments. The values are also dicts,
                   which index based on well, with the value being the
                   path of the experiment-well'''
        if exp != None:
            self.set_exp(exp)

        #Check if experiment already has been downloaded
        if os.path.isdir(self.data_path + self.exp):
            self.set_well_dict()

            #TODO: Check if this is going to cause an issue
            #!{s3} cp s3://{s3_path + exp + 'dataset/'} {data_path}{exp}
            print('Selected! Experiment already exists locally:',self.exp)
            return 

        print('Loading experiment: ' + self.exp[:-1])

        # Download files from exp
        cmd = f'{s3} cp s3://{s3_path + self.exp + res_path} {self.data_path + self.exp}. --recursive --exclude="*" --include="*.zip"'
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
#         !{s3} cp s3://{s3_path + exp + res_path} data/{exp}. --recursive --exclude="*" --include="*.zip"

        #Unzip those files
        cmd = f"unzip -qn {self.data_path + self.exp}*.zip -d {self.data_path + self.exp[:-1]}"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output, error = process.communicate()
        print(error)
#         !unzip -qn 'data/{exp}*.zip' -d data/{exp[:-1]}

        cmd = f"rm {self.data_path + self.exp}*.zip"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
        output, error = process.communicate()
#         !rm data/{exp}*.zip
        print(error)


        self.set_well_dict()

        print('Finished loading:',exp)
        return 

    def load_well(self,well=None):
        '''Loads corresponding wells neurons and ratings

        Arguments:
        well -- location of well (ex. 'A1')
        
        self.exp -- name of experiment followed by '/' (ex. test1/)

        Global vars:
        data_path -- path where data will be downloaded

        '''
        neurons = None

        #Sort by actual number
        well_data = {k: v for k,v in sorted(self.well_dict[well].items(),key=lambda x: x[0])}

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
        ratings_path = f'{self.data_path}{self.exp}dataset/'
        ratings_path = glob.glob(ratings_path + well + '*.npy')
        
        if self.exp in self.ratings_dict:
            ratings = load_ratings(self.ratings_dict[self.exp][well])
#             ratings = np.load(ratings_path[0])
            print('{} ratings loaded.'.format(len(ratings)))

        else:
            ratings = np.zeros(len(neurons))
            print('Ratings NOT loaded.')

        self.neurons = neurons
        self.ratings = ratings
        return (neurons,ratings)
    
    
    
    def gen_well_dropdown(self):
        '''
        Generates dropdown list for experiment selection

        Parameters:
        -----------
        exps: list
                list of experiments returned fro get_exp_list
                if None, generates list
        disp: bool
                if True, displays dropdown immediately
        '''        
        self.select_well = widgets.Dropdown( options=self.well_dict.keys(), description='Well:')

        
        display(self.select_well)
        return
    
    def gen_load_well_b(self):
        '''
        Generates load well button
        '''
        self.load_well_btn = widgets.Button(description="Load Well")
        self.load_well_btn.on_click(self.load_well_b)
        
        display(self.load_well_btn)
        return
        
        
    def load_well_b(self,b):
        '''Loads all channel groups from the specified well selected in the drop down menu'''

        #Load from dropdowns
        self.set_well()

        self.load_well(self.well)

        return
    
    
    def set_well(self,well=None):
        '''
        Sets well
        If well is none, try to pull from dropdown menu
        '''
        if well == None:
            self.well = self.select_well.value
        else:
            if self.select_well != None:
                self.select_well.value = well
            self.well = well
        
        return
    
    
    def rate_neuron_b(self,b):
        '''
        Rates current neuron, goes to next neuron/finishes
        '''
         
        #Rate neuron of i-1
        if type(b.description) != str:
            rate = int(b.description)
            self.ratings[self.ind_neurons]=rate


        #Show neuron i
    #     clear_output()
        clear_output(wait=True)
        self.ind_neurons = self.ind_neurons + 1

        #Finish if no more neurons
        if self.ind_neurons >= len(self.ratings):
            print('Finished!')
            return



        self.neurons[self.ind_neurons].qual_prob = [0,0,0,0]
        self.neurons[self.ind_neurons].checkqual(binsz=60)

        print('Neuron:{}/{}'.format(self.ind_neurons+1,self.n_neurons))
        print('Current Rating:',self.ratings[self.ind_neurons])
        display(self.btn_1,self.btn_2,self.btn_3,self.btn_4,self.btn_5)
        
        return
    
    def gen_rate_neuron(self):
        '''
        Iterates through list with buttons, rating neurons
        '''
        
        
        self.btn_1 = widgets.Button(description="1")
        self.btn_2 = widgets.Button(description="2")
        self.btn_3 = widgets.Button(description="3")
        self.btn_4 = widgets.Button(description="4")
        self.btn_5 = widgets.Button(description="Keep Current")
        
        self.btn_1.on_click(self.rate_neuron_b)
        self.btn_2.on_click(self.rate_neuron_b)
        self.btn_3.on_click(self.rate_neuron_b)
        self.btn_4.on_click(self.rate_neuron_b)
        self.btn_5.on_click(self.rate_neuron_b)
        
        self.n_neurons = len(self.neurons)
        self.ind_neurons = 0
        print('Starting rating for ratings/num_neurons:{}/{}'.format(len(self.ratings),self.n_neurons))
        
        self.neurons[self.ind_neurons].qual_prob = [0,0,0,0]
        self.neurons[self.ind_neurons].checkqual(binsz=10)
        print('Neuron:{}/{}'.format(self.ind_neurons+1,self.n_neurons))
        print('Current Rating:',self.ratings[self.ind_neurons])
        display(self.btn_1,self.btn_2,self.btn_3,self.btn_4, self.btn_5)
        
        
    def save_ratings(self,s3_upload=True):
        '''Saves the well ratings as a csv from the text input field prepended by the well.
        Uses selected experiment to save the well back into its corresponding dataset folder
        UPLOADS to s3 also'''

        save_dir = f'{self.data_path}{self.exp}dataset/'

        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir + self.well + '_' + self.name_field.value

        if not (os.path.isfile(save_path + '.npy')):
            np.save(save_path,np.array(self.ratings))
            print('Saved successfully in:',save_path)
        else:
            print("File already exists")

#         if b.description == "Save & Upload":
        if s3_upload:
            s3_save_path = self.exp + "dataset/" + self.well + '_' + self.name_field.value + '.npy'
            
            cmd = f"{s3} cp {save_path + '.npy'} s3://{s3_path + s3_save_path} "
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print('Uploaded Successfully')
#             !{s3} cp {save_path + '.npy'} s3://{s3_path + s3_save_path} 
    
    def save_ratings_b(self,b):
        
        if b.description == "Save & Upload":
            self.save_ratings()
        else:
            self.save_ratings(s3_upload=False)
        
    
    def gen_save_b(self):
        '''
        Generate save buttons and widgets for s3
        '''
        
        self.save_well_btn = widgets.Button(description="Save")
        self.save_well_s3_btn = widgets.Button(description="Save & Upload")
        self.name_field = widgets.Text(
            value='',
            placeholder='title here',
            description='Filename prepended by well (ex: "A1_YourTextHere")')

        self.save_well_btn.on_click(self.save_ratings_b)
        self.save_well_s3_btn.on_click(self.save_ratings_b)
        display(self.name_field,self.save_well_btn,self.save_well_s3_btn)
        return
    
def get_exp_list():
    '''
    Get experiment list from s3 location.

    '''
    cmd = f'{s3} ls s3://{s3_path}'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = str(output).split('\\n')
    exps = [t.split('PRE ')[1] for t in output if len(t.split('PRE ')) >1]
    return exps

def get_ratings_list():
    '''
    Get list of uuids which have datasets
    '''
    objs = wr.list_objects('s3://braingeneers/ephys/*/dataset/*.npy')
    
    datasets = []
    for o in objs:
        #This is dirty
        datasets.append(o.split('/')[4] + '/')

    return datasets


def get_ratings_dict():
    '''
    Returns dict of dict mapping UUID -> Well -> s3 path
    
    Usage of dict looks like d[uuid]['A1']
    
    '''
    objs = wr.list_objects('s3://braingeneers/ephys/*/dataset/*.npy')
    
    ratings_dict = {}
    
    for o in objs:
        #This is dirty
        key = o.split('/')[4] + '/'
        well = o.split('/')[6][:2]
        if ratings_dict.get(key) == None:
            ratings_dict[key] = {well:o}
        else:
            ratings_dict[key][well] = o

    return ratings_dict


def load_ratings(fname):
    '''
    Loads array of ratings from s3 filename
    
    Use get_ratings_dict to generate an easy method to index through the s3 filenames
    
    '''
    return NumpyS3Memmap(fname)[:]




def load_all_rated():
    '''
    Loads all neurons(outputted from the sorter) that have been rated
    '''
    na = NeuralAid()
    #Local storage
    na.set_data_path('./data/')
    
    ratings_dict = get_ratings_dict()
    
    neurons = []
    ratings = []
    for exp in ratings_dict.keys():
        for well in ratings_dict[exp].keys():

            na.load_experiment(exp)
            n_temp,r_temp = na.load_well(well)
            assert len(n_temp) == len(r_temp), "Number of neurons in the data and number of ratings must be the same, failure in {}".format(exp)
            neurons = np.append(neurons,n_temp)
            ratings = np.append(ratings,r_temp)

    print('Loaded {} neurons and ratings'.format(len(neurons)))
    return (neurons,ratings)
