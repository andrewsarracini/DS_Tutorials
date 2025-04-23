import argparse

def get_common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, default=7011, help='Target subject for LOSO')
    parser.add_argument('--model', type=str, default='lgbm', choices=['lgbm', 'rf'], help='Model selection')
    parser.add_argument('--trials', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('--last', action='store_true', help='Run the most recently added feature only')
    parser.add_argument('--baseline', action='store_true', help='Run only the baseline feature set')
    parser.add_argument('--allfeats', action='store_true', help='Run all features in the registry')

    return parser

def resolve_feats_to_run(all_features, args): 
    if args.baseline: 
        return [all_features[0]] 
    elif args.last:
        return [all_features[-1]]
    else: 
        return all_features