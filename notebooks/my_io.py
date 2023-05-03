import os
import mne
import pandas as pd

import struct
import time
import datetime as dt
import numpy as np
import mne
from mne.io import RawArray
from mne import create_info


def read_sef(filename):
    """
    Reads file with format .sef, and returns a mne.io.Raw object containing
    the data.

    Parameters
    ----------
    filename : str or file-like
        The Simple EEG (.sef) file to read.

    Returns
    -------
    raw : mne.io.RawArray
        RawArray containing the EEG signals.
    """
    f = open(filename, 'rb')
    #   Read fixed part of the headerà
    version = f.read(4).decode('utf-8')
    if version != 'SE01':
        print(f'Version : {version} not supported')
        raise ValueError()
    n_channels,         = struct.unpack('I', f.read(4))
    num_aux_electrodes, = struct.unpack('I', f.read(4))
    num_time_frames,    = struct.unpack('I', f.read(4))
    sfreq,              = struct.unpack('f', f.read(4))
    year,               = struct.unpack('H', f.read(2))
    month,              = struct.unpack('H', f.read(2))
    day,                = struct.unpack('H', f.read(2))
    hour,               = struct.unpack('H', f.read(2))
    minute,             = struct.unpack('H', f.read(2))
    second,             = struct.unpack('H', f.read(2))
    millisecond,        = struct.unpack('H', f.read(2))

    #   Read variable part of the header
    ch_names = []
    for _ in range(n_channels):
        name = [char for char in f.read(8).split(b'\x00')
                if char != b''][0]
        ch_names.append(name.decode('utf-8').strip())
    # Read data
    buffer = np.frombuffer(
        f.read(n_channels * num_time_frames * 8),
        dtype=np.float32,
        count=n_channels * num_time_frames)
    data = np.reshape(buffer, (num_time_frames, n_channels))
    # Create infos
    description = 'Imported with Pycartool'
    try:
        record_time = dt.datetime(year, month, day,
                                  hour, minute, second).timetuple()
        meas_date = (time.mktime(record_time), millisecond)
    except Exception as e:
        print('Cannot read recording date from file...')
        print(e)
        meas_date = None
    ch_types = ['eeg' for i in range(n_channels)]
    infos = create_info(ch_names=ch_names, sfreq=sfreq,
                        ch_types=ch_types)
    infos['description'] = description
    raw = RawArray(np.transpose(data), infos)
    raw.set_meas_date(meas_date)
    return (raw)



def read_bad_file(path, sfreq):
    df = pd.read_csv(path, sep="\t", skiprows=1, names=['start', 'stop', 'label'])
    df['start_time'] = df['start'] / sfreq
    df['stop_time'] = df['stop'] / sfreq
    df['duration'] = df['stop_time'] - df['start_time']
    bads_annotations = mne.Annotations(df['start_time'], df['duration'], df['label'])
    return(bads_annotations)

def read_epileptic_events_file(path, sfreq):
    df = pd.read_csv(path, sep="\t", skiprows=1, names=['start', 'stop', 'label'])
    df['start_time'] = df['start'] / sfreq
    df['stop_time'] = df['stop'] / sfreq
    df['duration'] = df['stop_time'] - df['start_time']

    events = list()
    for r, row in df.iterrows():
        label = row['label']
        if label.lower() == 'hpd_start':
            start = row['start_time']
            for r_, row_ in df.iloc[r:].iterrows():
                label_ = row_['label']
                if label_.lower() == 'hpd_end':
                    stop = row_['start_time']
                    event = dict()
                    event['label'] = 'hpd'
                    event['start'] = start
                    event['stop'] = stop
                    event['duration'] = stop - start
                    events.append(event)
                    break

    df_annot = pd.DataFrame(events)
    epileptic_annotations = mne.Annotations(df_annot['start'], df_annot['duration'], df_annot['label'])
    return(epileptic_annotations)


def read_background_events_file(path, sfreq, window_size=1):
    df = pd.read_csv(path, sep="\t", skiprows=1, names=['start', 'stop', 'label'])
    df['start_time'] = df['start'] / sfreq - (window_size/2)
    df['duration'] = window_size # 1sec of background around marker
    background_annotations = mne.Annotations(df['start_time'], df['duration'], 'background')
    return(background_annotations)


def read_file(fname):
    # Read Raw
    base_path = os.path.dirname(fname) 
    raw = read_sef(fname)
    # Read Bads
    bad_annotations = mne.Annotations(0, 0, 'null')
    for file in os.listdir(base_path):
        if file.lower().startswith('bad'):
            print(file)
            path = os.path.join(base_path, file)
            annotations = read_bad_file(path, raw.info['sfreq'])
            bad_annotations += annotations
    # Read epileptic
    epileptic_annotations = mne.Annotations(0, 0, 'null')
    for file in os.listdir(base_path):
        if file.lower().startswith('epileptic'):
            print(file)
            path = os.path.join(base_path, file)
            annotations = read_epileptic_events_file(path, raw.info['sfreq'])
            epileptic_annotations += annotations
    # Read background
    background_annotations = mne.Annotations(0, 0, 'null')
    for file in os.listdir(base_path):
        if file.lower().endswith('bck.mrk'):
            print(file)
            path = os.path.join(base_path, file)
            annotations = read_background_events_file(path, raw.info['sfreq'])
            background_annotations += annotations
    annotations = epileptic_annotations + bad_annotations + background_annotations
    raw.set_annotations(annotations) 
    return(raw)