# scripts/stream_sim_runner.py

import argparse
from pathlib import Path
from sleep_wave.stream.simulator import StreamSim
from sleep_wave.utils.file_resolver import get_edf_paths

def main(edf_path, hypnogram_path, max_epochs):
    stream = StreamSim(
        edf_path=edf_path,
        hypnogram_path=hypnogram_path,
        extract_features=True,
        return_labels=True
    )

    print(f"\n🚀 Starting simulated EEG stream from: {edf_path.name}")
    for i, epoch in enumerate(stream):
        print(f"\n--- Epoch {i + 1} ---")
        print(f"Subject ID: {epoch['subject_id']}")
        print(f"Sample Index: {epoch['sample_index']}")
        print(f"EEG Shape: {epoch['eeg'].shape}")

        if stream.extract_features:
            print("Features:")
            for k, v in epoch['features'].items():
                print(f"  {k}: {v:.4f}")

        if stream.return_labels:
            print(f"Label: {epoch.get('label', 'None')}")

        if i + 1 >= max_epochs:
            print("\n✅ Stream simulation complete.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time EEG stream from EDF")
    parser.add_argument("--subject", type=str, help="4-digit subject ID (e.g., 7011)")
    parser.add_argument("--edf", type=str, help="Optional: Path to PSG EDF file (overrides subject logic)")
    parser.add_argument("--hypnogram", type=str, help="Optional: Path to Hypnogram EDF file")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs to simulate")

    args = parser.parse_args()

    # Path resolution logic
    if args.edf:
        edf_path = Path(args.edf)
        hypnogram_path = Path(args.hypnogram) if args.hypnogram else None
    elif args.subject:
        edf_path, hypnogram_path = get_edf_paths(args.subject)
    else:
        raise ValueError("You must specify either --subject or both --edf and --hypnogram.")

    main(
        edf_path=edf_path,
        hypnogram_path=hypnogram_path,
        max_epochs=args.max_epochs
    )