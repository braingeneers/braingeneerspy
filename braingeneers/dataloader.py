import datasets_electrophysiology as de
import time

# dataset class to use for dataloader
class EphysDataset(Dataset):
    def __init__(self, batch_uuid, experiment_num):
        dataset = de.load_blocks(batch_uuid, experiment_num)
        self.length = dataset.shape[0]
        # self.x are all the channels in the dataset, use torch.from_numpy to make a tensor
        self.x = torch.from_numpy(dataset[:, 1:])


    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.length

# set a small range of frames to take
dataset = EphysDataset('2021-10-05-e-org1_real', 0)
myDataloader = Dataloader(dataset=dataset, shuffle=True)
rando = torch.random()
t0 = time.time()
for data in myDataloader:

    print(f'I got this data in {time.time() - t0} seconds and sample size was {data.shape}')
    t0 = time.time()
