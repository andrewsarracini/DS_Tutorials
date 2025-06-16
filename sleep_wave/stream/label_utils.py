# sleep_wave/stream/label_utils.py

def extract_epoch_labels(annotations, epoch_len, total_duration, stage_map):
    n_epochs = total_duration // epoch_len
    labels = ['UNKNOWN'] * n_epochs
    for ann in annotations:
        label = stage_map.get(ann['description'], None)
        if label is None:
            continue
        start = int(ann['onset']) // epoch_len
        duration = int(ann['duration']) // epoch_len
        for i in range(start, min(start + duration, n_epochs)):
            labels[i] = label
    return labels
