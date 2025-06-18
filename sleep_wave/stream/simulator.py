import numpy as np
import mne
from mne.time_frequency import psd_array_welch

from sleep_wave.features.utils import bandpower
from sleep_wave.stream.label_utils import extract_epoch_labels


class StreamSim:
    def __init__(self, edf_path, hypnogram_path=None, epoch_len=30,
                 extract_features=False, return_labels=False,
                 skip_unlabeled_head=True, debug=False):
        """
        Simulates real-time EEG streaming by yielding one 30s epoch at a time.

        Parameters:
            edf_path (Path or str): Path to PSG (EEG) EDF file
            hypnogram_path (Path or str): Path to hypnogram EDF file (for labels)
            epoch_len (int): Length of each simulated epoch in seconds
            extract_features (bool): Whether to compute bandpower features
            return_labels (bool): Whether to include true labels from hypnogram
            skip_unlabeled_head (bool): If True, skips epochs at the beginning with 'UNKNOWN' label
            debug (bool): If True, prints debug info
        """

        self.raw = mne.io.read_raw_edf(edf_path, preload=True)
        if hypnogram_path and return_labels:
            self.raw.set_annotations(mne.read_annotations(hypnogram_path))

        self.raw.pick(['EEG Fpz-Cz'])
        self.sfreq = self.raw.info['sfreq']
        self.data = self.raw.get_data()[0]
        self.epoch_len = epoch_len
        self.samples_per_epoch = int(epoch_len * self.sfreq)
        self.total_samples = len(self.data)
        self.extract_features = extract_features
        self.return_labels = return_labels
        self.subject_id = str(edf_path.name)[2:6]  # Assumes ST####J format

        # Label normalization map
        self.label_map = {
            'sleep stage w': 'Wake',
            'sleep stage 1': 'N1',
            'sleep stage 2': 'N2',
            'sleep stage 3': 'N3',
            'sleep stage 4': 'N3',
            'sleep stage r': 'REM'
        }

        self.labels = []
        if hypnogram_path and return_labels:
            total_secs = self.total_samples / self.sfreq
            self.labels = extract_epoch_labels(
                self.raw.annotations,
                epoch_len,
                int(total_secs),
                self.label_map,
                debug=debug
            )

        # Determine starting index
        if return_labels and skip_unlabeled_head and self.labels:
            try:
                self.start_idx = next(i for i, lbl in enumerate(self.labels) if lbl != 'UNKNOWN')
            except StopIteration:
                raise ValueError("No labeled epochs found.")
        else:
            self.start_idx = 0

        self.current_idx = self.start_idx

    def __iter__(self):
        return self

    def __next__(self):
        start = self.current_idx * self.samples_per_epoch
        end = start + self.samples_per_epoch
        if end > self.total_samples:
            raise StopIteration

        segment = self.data[start:end]
        self.current_idx += 1

        out = {
            "eeg": segment,
            "subject_id": self.subject_id,
            "sample_index": start
        }

        if self.extract_features:
            psds, freqs = psd_array_welch(
                segment[np.newaxis, np.newaxis, :],
                sfreq=self.sfreq,
                fmin=0.5,
                fmax=40,
                n_fft=self.samples_per_epoch
            )
            out["features"] = {
                'delta': bandpower(psds, freqs, (0.5, 4))[0],
                'theta': bandpower(psds, freqs, (4, 8))[0],
                'alpha': bandpower(psds, freqs, (8, 13))[0],
                'beta':  bandpower(psds, freqs, (13, 30))[0]
            }

        if self.return_labels and self.current_idx - 1 < len(self.labels):
            out["label"] = self.labels[self.current_idx - 1]

        return out
