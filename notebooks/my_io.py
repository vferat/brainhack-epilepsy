import os
import mne
import pandas as pd
import pycartool


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
    base_name = os.path.basename(fname)
    raw = pycartool.io.read_sef(fname)
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