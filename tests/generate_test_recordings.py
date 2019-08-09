"""
Generate test raw recording data
"""
import struct
import datetime
import pytz
import numpy as np


def ntp_to_datetime(ts):
    """ Convert Internet Network Time Protocol (NTP) 64bit timestamp into python datetime """
    seconds = int(ts) >> 32
    microseconds = np.bitwise_and(ts, np.uint64(0xFFFFFFFF)) / 2**32
    return (datetime.timedelta(seconds=seconds+microseconds) +
            datetime.datetime(1900, 1, 1, 0, 0, 0, 0, pytz.utc))


def datetime_to_ntp(ts):
    """ Convert python datetime into an Internet Network Time Protocol (NTP) 64bit timestamp """
    diff = ts - datetime.datetime(1900, 1, 1, 0, 0, 0, 0, pytz.utc)
    return np.uint64((int(diff.total_seconds()) << 32) + (diff.microseconds / 1000000 * 2**32))


if __name__ == '__main__':
    # Test round trip from python to NTP to python timestamps
    datetime_ts = datetime.datetime(2019, 2, 5, 19, 38, 42, 314159,
                                    datetime.timezone(-datetime.timedelta(hours=8)))
    ntp_ts = datetime_to_ntp(datetime_ts)
    assert datetime_ts == ntp_to_datetime(ntp_ts)

    # 1.5 minutes, 4 channels @ 1Hz sample rate
    X = np.array([range(i*10, i*10+4) for i in range(60 + 30)], dtype=np.uint16)
    print("Generated recording with shape", X.shape)

    # Write out two blocks, one a second long and one a half second
    with open("derived/test-datasets/experiment_190205_193842.bin", "wb") as f:
        ts = datetime.datetime(2019, 2, 5, 19, 38, 42, 314159,
                               datetime.timezone(-datetime.timedelta(hours=8)))
        print(ts.isoformat())
        f.write(struct.pack('Q', datetime_to_ntp(ts)))
        print(bin(datetime_to_ntp(ts)))
        print(datetime_to_ntp(ts))
        X[0:60, :].astype('uint16').tofile(f)
        print(X[0:60, :].astype('uint16'))

    with open("derived/test-datasets/experiment_190205_193942.bin", "wb") as f:
        ts = datetime.datetime(2019, 2, 5, 19, 39, 42, 314159,
                               datetime.timezone(-datetime.timedelta(hours=8)))
        print(ts.isoformat())
        f.write(struct.pack('Q', datetime_to_ntp(ts)))
        print(bin(datetime_to_ntp(ts)))
        X[60:, :].astype('uint16').tofile(f)
        print(X[60:, :].astype('uint16'))

    with open("derived/test-datasets/experiment_190205_193842.bin", "rb") as f:
        ts = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
        raw = np.fromfile(f, dtype=np.uint16)

    num_channels = 4

    ntp_to_datetime(ts).isoformat()
    print(raw.reshape(-1, num_channels, order="C"))
