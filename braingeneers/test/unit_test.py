# from braingeneers.utils import s3wrangler as sw
# from braingeneers.utils import smart_open as so
from braingeneers import datasets_electrophysiology as de
import numpy as np
from braingeneers.ml import ephys_dataloader as ep
from torch.utils.data import DataLoader
import time

# import io

wrap = {
    "blocks": [
        {
            "num_frames": 8037520800,
            "path": "/home/mxwbio/Data/omfg/OMFG_FORREALTHO.raw.h5",
            "source": "",
            "timestamp": "2021-10-05T17:03:55"
        },
        {
            "num_frames": 8037520800,
            "path": "/home/mxwbio/Data/omfg/OMFG_FORREALTHO.raw.h5",
            "source": "",
            "timestamp": "2021-10-05T17:03:55"
        }
    ],
    "channels": [],
    "hardware": "Maxwell",
    "name": "1well-maxwell",
    "notes": "",
    "num_channels": 1028,
    "num_current_input_channels": 0,
    "num_voltage_channels": 1028,
    "offset": 0,
    "sample_rate": 20000,
    "voltage_scaling_factor": 1,
    "timestamp": "2021-10-05T17:03:55",
    "units": "\u00b5V",
    "version": "0.0.1"
}

'''
block_dict = {i: wrap['blocks'][i]['num_frames']/wrap['num_channels'] for i in range(len(wrap['blocks']))}
print(block_dict)
'''
'''
a = sum([s["num_frames"] for s in wrap['blocks'][0:1]])
print (a)
'''

'''
myfile = io.BytesIO() # io.BytesIO wraps the byte array into a file-like object
sw.download('s3://braingeneers/ephys/2021-10-05-e-org1_real/metadata.json', myfile)
print(myfile.getvalue())

myfile2 = so.open('s3://braingeneers/ephys/2021-10-05-e-org1_real/metadata.json', 'r')
print(myfile2.read())
'''

# first load the batch of all experiments matching the UUID. The batch has a format of a dictionary.
'''
exp_batch = de.load_batch('2021-10-05-e-org1_real')
print(exp_batch)
# now, grab an experiment using load_experiment. This will return all the data associated with the experiment.
exp_load = de.load_experiment('2021-10-05-e-org1_real', 0)

print(exp_load)
'''
# print(exp_load['blocks'][0]['path'].split('/')[-1])

# testing load_blocks into load_maxwell
# make sure to pass small values to test functionality
'''
X, t, fs = de.load_blocks(batch_uuid='2021-10-05-e-org1_real', experiment_num=0, channels=[0, 1, 2])
print('X: ', X,'t: ', t,'fs: ', fs)
'''

# the final form should be something like
'''
data_loader = iter(MyDataLoader(...))
for data in data_loader:
   print('I got this sample in N seconds, and sample size was XYZ')
'''
# Would be a good idea to wrap the dataset in a subclass so that the dataLoader can

'''
# testing iteration
block = 1
total_frames = 0
for b in range(0, block):
    total_frames += wrap['blocks'][b]['num_frames']
print(total_frames)

print(wrap['blocks'][block]['timestamp'].split('T')[-1])
'''
'''
dataset, fs, num_frames = de.load_data(batch_uuid='2021-10-05-e-org1_real', experiment_num=0, channels=[0, 1, 2],
                                       offset=20, length=15000)
print(' dataset: ', dataset, ' sample rate: ', fs, ' number of frames: ', num_frames)
print(dataset.shape)
print(de.compute_milliseconds(num_frames, fs))
'''
'''
def getrange(array,idx):
    chunk = array[:, idx-7:idx+1]
    return  chunk
some_data = np.array(([1,1,1], [2,2,2], [3,3,3]))
tester = np.empty((3,3))
print(tester, some_data)
tester[:, 0:1] = some_data[:, 0:1]
print(tester, some_data)
tester[:, 0:1] = 0
print(tester)
'''
dataset = ep.EphysDataset(batch_uuid='2021-10-05-e-org1_real', experiment_num=0, sample_size=5000, bounds='pad', channels=[0],
                          offset=6000, length=10000)
myDataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
# rando = torch.random()
t0 = time.time()
for data in myDataloader:
    print(f'I got this data in {time.time() - t0} seconds and sample size was {data.shape}')
    t0 = time.time()
