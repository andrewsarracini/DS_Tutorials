import argparse
from src.helper import eval_best_config
# Add more tools later!

def main():
    parser = argparse.ArgumentParser(description='Sleepwave CLI Tool')
    subparser = parser.add_subparsers(dest='command') 

    # Subcommand: eval_best_config
    eval_parser = subparser.add_parser('eval', help='Evaluate saved best configs on a target subject')
    eval_parser.add_argument('--config-path', type=str, required=True, help='Path to saved config JSON')
    eval_parser.add_argument('--subject', type=int, required=True, help='Subject ID to eval on')

    args = parser.parse_args()

    if args.command == 'eval':
        eval_best_config(args.config_path, args.subject)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
    