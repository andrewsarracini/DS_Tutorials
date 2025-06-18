# sleep_wave/stream/label_utils.py

def extract_epoch_labels(annotations, epoch_len, total_duration, stage_map):
    n_epochs = total_duration // epoch_len
    labels = ['UNKNOWN'] * n_epochs

    try: 
        # try treating as mne.Annotations
        iterator = zip(
            annotations.onset,
            annotations.duration, 
            annotations.description
        )
    except AttributeError:
        # Fallback to OrderedDict-style annotations
        iterator = (
            (ann['onset'], ann['duration'], ann['description']) for ann in annotations
        )
    
    for onset, duration, description in iterator:
        normalized = description.strip().lower()
        label = stage_map.get(normalized)
        if label is None: 
            continue

        start = int(onset) // epoch_len
        duration_epochs = int(duration) // epoch_len
        
        for i in range(start, min(start + duration_epochs, n_epochs)):
            labels[i] = label

    return labels
            