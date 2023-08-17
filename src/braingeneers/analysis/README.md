# analysis package

The `analysis` module provides utilities for analyzing neuronal spike data, including loading, filtering, detecting spikes, and calculating metrics.

## Loading Data

```python
load_spike_data(uuid, experiment=None, basepath=None)
```

Loads spike data from a dataset given the dataset's UUID, experiment name, and optional basepath. Returns a `SpikeData` object containing the spike times, neuron attributes, and metadata.

```python
read_phy_files(path) 
```

Loads spike data from a zip file containing Phy output, returning a `SpikeData` object.

## Filtering

```python
filter(raw_data, fs_Hz=20000, filter_order=3, 
       filter_lo_Hz=300, filter_hi_Hz=6000, time_step_size_s=10)
```

Applies a bandpass Butterworth filter to raw data.

## Spike Detection

```python
ThresholdedSpikeData(raw_data, ...)
```

The `ThresholdedSpikeData` class detects spikes in raw data by thresholding and optional hysteresis.

## Metrics

```python
fano_factors(raster)
pearson(spikes)
cumulative_moving_average(hist)
burst_detection(spike_times, ...) 
```

Functions to calculate metrics on `SpikeData`.

```python
deviation_from_criticality(...)
``` 

Calculates the deviation from criticality metric according to Ma et al. 2019. Returns a `DCCResult` namedtuple.

## Spike Data Manipulation

```python
SpikeData(arg1, arg2=None, *)
```

The `SpikeData` class represents neuronal spike data and metadata. Contains methods to iterate over spikes, bin spikes, calculate rates, and more.

```python
randomize_raster(raster)
```

Randomizes a spike raster while preserving firing rates. 

```python
best_effort_sample(counts, M) 
```

Samples from a discrete distribution without replacement if possible.