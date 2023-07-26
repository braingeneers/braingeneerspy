from torch.utils.data import Dataset
from braingeneers.data import datasets_electrophysiology as de
import numpy as np
import pandas as pd


# HERE set numpy seed
# todo: Need to have a good default for experiment_num and sample_size so can be used w/ spectrogram easily

class EphysDataset(Dataset):

    def __init__(self, batch_uuid, experiment_num, sample_size, attr_csv, align='center', bounds='exception',
                 length=-1,
                 channels=None):
        """
        :param batch_uuid: str
            String indicating which batch to take from
        :param experiment_num: int
            Number of desired experiment
        :param sample_size: int
            This value should be passed in every time; this determines the size of the samples the dataloader
                            should be returning. This is NOT the size of the dataset being loaded in by load_data.
        :param attr_csv: str of filepath or name of readable file
            CSV file specifying offset, length, and channels. Offset MUST be specified.
        :param align: str
            Indicates how to use idx: indicating center of data datachunk, left of datachunk, or right of datachunk
        :param bounds: str
            Indicates how to treat cases where there's OutOfBounds issues (e.g. center idx @ 0 would go below 0). options are
            'exception', 'pad', 'flush'.
        """
        self.attr_df = pd.read_csv(attr_csv)
        # store as self variables to do
        self.UUID = batch_uuid
        self.exp_num = experiment_num
        self.datalen = length
        self.channels = channels
        self.align = align
        self.bounds = bounds
        self.sample_size = sample_size
        # self.x are all the channels in the dataset, use torch.from_numpy to make a tensor
        # since pytorch is channels first, dataset doesn't need to do .transpose to work with from_numpy
        # self.x = torch.from_numpy(dataset)

    def __getitem__(self, idx):
        # should return a datachunk of frames, not each channel
        # empty array is rows = channels, col = sample size

        # indexing row of csv
        self.offset = self.attr_df.iloc[idx][0]
        data_length = self.attr_df.iloc[idx][1]
        if data_length is not None:
            self.datalen = data_length
        channels = self.attr_df.iloc[idx][2]
        # checking if 'all' option was given. should be true if 'all'.
        if any(char.isalpha() for char in channels):
            self.channels = [i for i in range(0, 1028)]
        # otherwise, use number passed in
        else:
            self.channels = [int(i) for i in channels.split('/')]

        dataset = de.load_data(de.load_metadata(self.UUID), self.exp_num, self.offset, self.datalen, self.channels)
        datachunk = np.empty((dataset.shape[0], self.sample_size))
        # if 'center', idx should point to the CENTER of a set of data, and get the frames from halfway in front and behind.
        # 4/4/22 replacing idx with offset. idx now points to a certain row in the csv.
        # offset now refers to the point in data where we're interested in sampling from.
        if self.align == 'center':
            left_bound = int(self.offset - (self.sample_size / 2))
            right_bound = int(self.offset + (self.sample_size / 2))
            # checking bounds handling
            if self.bounds == 'exception':
                # exception means throw an exception regarding usage
                if left_bound < 0 or right_bound > dataset.shape[1]:
                    raise IndexError('Sample size too large in one direction. Please use valid index/sample size pair.')
                else:
                    datachunk = dataset[:, left_bound:right_bound]
            elif self.bounds == 'pad':
                # pad means we fill the empty space with zeros
                if left_bound < 0 and right_bound > dataset.shape[1]:
                    # perform both adjustments
                    left_pad = 0 - left_bound
                    left_bound = 0
                    right_pad = right_bound - dataset.shape[1]
                    # pad both sides
                    datachunk[:, :left_pad] = 0
                    datachunk[:, right_pad:] = 0
                    datachunk[:, left_pad:right_pad] = dataset[:, left_bound:right_pad - left_pad]
                elif left_bound < 0:
                    # find the dimensions of padding space and reset left_bound, pad with 0s and fill rest of datachunk
                    left_pad = 0 - left_bound
                    left_bound = 0
                    datachunk[:, :left_pad] = 0
                    datachunk[:, left_pad:] = dataset[:, left_bound:right_bound]
                elif right_bound > dataset.shape[1]:
                    # calc overshoot and reset right_bound, fill with datachunk first and pad with 0s
                    right_bound = dataset.shape[1]
                    datalimit = right_bound - left_bound
                    datachunk[:, :datalimit] = dataset[:, left_bound:right_bound]
                    datachunk[:, datalimit:] = 0
                else:
                    # if none of these apply, then the bounds are valid and should be used.
                    datachunk = dataset[:, left_bound:right_bound]
            elif self.bounds == 'flush':
                # if it's flush, then make adjustment to the nearest viable sample.
                # if the sample size cannot be accommodated, shrink the bounds arbitrarily.
                if left_bound < 0 and right_bound > dataset.shape[1]:
                    left_bound = 0
                    right_bound = dataset.shape[1]
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                elif left_bound < 0:
                    left_bound = 0
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                elif right_bound > dataset.shape[1]:
                    right_bound = dataset.shape[1]
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                else:
                    # if none of these apply, then the bounds are valid and should be used.
                    datachunk = dataset[:, left_bound:right_bound]
        # if 'left', idx should be on left side of data, sampling forward
        elif self.align == 'left':
            # get bounds
            left_bound = self.offset
            right_bound = self.offset + self.sample_size
            if self.bounds == 'exception':
                # exception means throw an exception regarding usage
                if left_bound < 0 or right_bound > dataset.shape[1]:
                    raise IndexError('Sample size too large in one direction. Please use valid index/sample size pair.')
                else:
                    datachunk = dataset[:, left_bound:right_bound]
            elif self.bounds == 'pad':
                # pad means we fill the empty space with zeros
                if left_bound < 0 and right_bound > dataset.shape[1]:
                    # perform both adjustments
                    left_pad = 0 - left_bound
                    left_bound = 0
                    right_pad = right_bound - dataset.shape[1]
                    # pad both sides
                    datachunk[:, :left_pad] = 0
                    datachunk[:, right_pad:] = 0
                    datachunk[:, left_pad:right_pad] = dataset[:, left_bound:right_pad - left_pad]
                elif left_bound < 0:
                    # find the dimensions of padding space and reset left_bound, pad with 0s and fill rest of datachunk
                    left_pad = 0 - left_bound
                    left_bound = 0
                    datachunk[:, :left_pad] = 0
                    datachunk[:, left_pad:] = dataset[:, left_bound:right_bound]
                elif right_bound > dataset.shape[1]:
                    # calc overshoot and reset right_bound, fill with datachunk first and pad with 0ss
                    right_pad = right_bound - dataset.shape[1]
                    right_bound = dataset.shape[1]
                    datachunk[:, :right_pad] = dataset[:, left_bound:right_bound]
                    datachunk[:, right_pad:] = 0
                else:
                    # if none of these apply, then the bounds are valid and should be used.
                    datachunk = dataset[:, left_bound:right_bound]
            elif self.bounds == 'flush':
                # if it's flush, then make adjustment to the nearest viable sample.
                # if the sample size cannot be accommodated, shrink the bounds arbitrarily.
                if left_bound < 0 and right_bound > dataset.shape[1]:
                    left_bound = 0
                    right_bound = dataset.shape[1]
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                elif left_bound < 0:
                    left_bound = 0
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                elif right_bound > dataset.shape[1]:
                    right_bound = dataset.shape[1]
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                else:
                    # if none of these apply, then the bounds are valid and should be used.
                    datachunk = dataset[:, left_bound:right_bound]
            #datachunk = dataset[:, idx:idx + self.sample_size]
        # if 'right', idx is on the right end of the data, sampling backwards
        elif self.align == 'right':
            left_bound = self.offset - self.sample_size
            right_bound = self.offset
            if self.bounds == 'exception':
                # exception means throw an exception regarding usage
                if left_bound < 0 or right_bound > dataset.shape[1]:
                    raise IndexError('Sample size too large in one direction. Please use valid index/sample size pair.')
                else:
                    datachunk = dataset[:, left_bound:right_bound]
            elif self.bounds == 'pad':
                # pad means we fill the empty space with zeros
                if left_bound < 0 and right_bound > dataset.shape[1]:
                    # perform both adjustments
                    left_pad = 0 - left_bound
                    left_bound = 0
                    right_pad = right_bound - dataset.shape[1]
                    # pad both sides
                    datachunk[:, :left_pad] = 0
                    datachunk[:, right_pad:] = 0
                    datachunk[:, left_pad:right_pad] = dataset[:, left_bound:right_pad - left_pad]
                elif left_bound < 0:
                    # find the dimensions of padding space and reset left_bound, pad with 0s and fill rest of datachunk
                    left_pad = 0 - left_bound
                    left_bound = 0
                    datachunk[:, :left_pad] = 0
                    datachunk[:, left_pad:] = dataset[:, left_bound:right_bound]
                elif right_bound > dataset.shape[1]:
                    # calc overshoot and reset right_bound, fill with datachunk first and pad with 0ss
                    right_pad = right_bound - dataset.shape[1]
                    right_bound = dataset.shape[1]
                    datachunk[:, :right_pad] = dataset[:, left_bound:right_bound]
                    datachunk[:, right_pad:] = 0
                else:
                    # if none of these apply, then the bounds are valid and should be used.
                    datachunk = dataset[:, left_bound:right_bound]
            elif self.bounds == 'flush':
                # if it's flush, then make adjustment to the nearest viable sample.
                # if the sample size cannot be accommodated, shrink the bounds arbitrarily.
                if left_bound < 0 and right_bound > dataset.shape[1]:
                    left_bound = 0
                    right_bound = dataset.shape[1]
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                elif left_bound < 0:
                    left_bound = 0
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                elif right_bound > dataset.shape[1]:
                    right_bound = dataset.shape[1]
                    datachunk[:, left_bound:right_bound] = dataset[:, left_bound:right_bound]
                else:
                    # if none of these apply, then the bounds are valid and should be used.
                    datachunk = dataset[:, left_bound:right_bound]
        return datachunk

    def __len__(self):
        # change to reflect length of the csv, NOT the datapoints
        return len(self.attr_df.index)
