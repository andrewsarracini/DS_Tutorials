import subprocess
import argparse

from sleep_wave.features.registry import register_all_features

def run_cli(subject, model, feature_index=None, trials=10, baseline=False):
    '''
    Run the CLI command to execute the experiment with specified params
    '''
    