# sleep_wave/stream/label_utils.py

def extract_epoch_labels(annotations, epoch_len, total_duration, stage_map):
    n_epochs = total_duration // epoch_len
    labels = ['UNKNOWN'] * n_epochs

    for ann in annotations:
        normalized = ann['description'].strip().lower()
        label = stage_map.get(normalized)
        if label is None:
            continue
        start = int(ann['onset']) // epoch_len
        duration_epochs = int(ann['duration']) // epoch_len
        for i in range(start, min(start + duration_epochs, n_epochs)):
            labels[i] = label

        if label is None:
            print(f"[WARN] Unmapped label: '{ann['description']}' (normalized: '{normalized}')")
            continue


    return labels
