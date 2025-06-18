from collections import Counter

def extract_epoch_labels(annotations, epoch_len, total_duration, stage_map, debug=False):
    n_epochs = total_duration // epoch_len
    labels = ['UNKNOWN'] * n_epochs

    raw_counts = Counter()

    for ann in annotations:
        raw = ann['description']
        normalized = raw.strip().lower()
        label = stage_map.get(normalized, None)
        raw_counts[normalized] += 1

        if debug:
            print(f"[ANNOTATION] '{raw}' → normalized: '{normalized}'")

        if label is None:
            if debug:
                print(f"  [UNMAPPED] '{normalized}' not in stage_map")
            continue

        onset_epoch = int(ann['onset']) // epoch_len
        duration_epochs = int(ann['duration']) // epoch_len
        end_epoch = min(onset_epoch + duration_epochs, n_epochs)

        if debug:
            print(f"  [MAPPED] '{normalized}' → '{label}'")
            print(f"  → Labeling epochs {onset_epoch} to {end_epoch - 1} with '{label}'")

        for i in range(onset_epoch, end_epoch):
            labels[i] = label

    if debug:
        print(f"[SUMMARY] Raw Label Counts:\n{raw_counts}")
        print(f"[SUMMARY] Final Label Distribution: {Counter(labels)}")

    return labels
