import math

def extract_epoch_labels(annotations, epoch_len, total_duration, stage_map):
    n_epochs = int(math.ceil(total_duration / epoch_len))
    labels = ['UNKNOWN'] * n_epochs

    for ann in annotations:
        desc = ann['description'].strip().lower()
        print(f"[NORMALIZED CHECK] '{ann['description']}' → '{desc}'")
        label = stage_map.get(desc)
        if label is None:
            print(f"[UNMAPPED] '{desc}'")
        else: 
            print(f'[MAPPED] "{desc}" -> {label}')
            continue

        onset = float(ann['onset'])
        duration = float(ann['duration'])

        start_epoch = int(onset // epoch_len)
        end_epoch = int(math.ceil((onset + duration) / epoch_len))

        for i in range(start_epoch, min(end_epoch, n_epochs)):
            labels[i] = label

    return labels
