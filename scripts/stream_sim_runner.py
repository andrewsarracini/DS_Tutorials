# scripts/stream_sim_runner.py

import argparse
from pathlib import Path
from sleep_wave.stream.simulator import StreamSim

def main(edf_path, hypnogram_path=None, extract_features=False, return_labels=False, max_epochs=5):
    stream = StreamSim(
        edf_path=Path(edf_path),
        hypnogram_path=Path(hypnogram_path) if hypnogram_path else None,
        extract_features=extract_features,
        return_labels=return_labels
    )

    print(f"\n🚀 Starting stream from: {edf_path}")
    for i, epoch in enumerate(stream):
        print(f"\n--- Epoch {i + 1} ---")
        print(f"Subject ID: {epoch['subject_id']}")
        print(f"Sample Index: {epoch['sample_index']}")
        print(f"EEG Shape: {epoch['eeg'].shape}")

        if extract_features:
            print("Features:")
            for k, v in epoch['features'].items():
                print(f"  {k}: {v:.4f}")

        if return_labels:
            print(f"Label: {epoch.get('label', 'None')}")

        if i + 1 >= max_epochs:
            print("\n✅ Stream simulation complete.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate EEG streaming from EDF")
    parser.add_argument("--edf", required=True, help="Path to EEG EDF file (e.g., PSG.edf)")
    parser.add_argument("--hypnogram", help="Path to hypnogram EDF (optional)")
    parser.add_argument("--extract_features", action="store_true", help="Enable bandpower feature extraction")
    parser.add_argument("--return_labels", action="store_true", help="Include true labels from hypnogram")
    parser.add_argument("--max_epochs", type=int, default=5, help="Max epochs to simulate")

    args = parser.parse_args()
    main(
        edf_path=args.edf,
        hypnogram_path=args.hypnogram,
        extract_features=args.extract_features,
        return_labels=args.return_labels,
        max_epochs=args.max_epochs
    )
