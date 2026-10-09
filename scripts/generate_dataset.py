"""Prepare or convert a dataset described by a generation YAML."""
import argparse
import json

from hydra import compose, initialize_config_dir

from connectpt.routes_generator.run.const import CFG_DIR
from connectpt.routes_generator.run.dataset import prepare_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name',required=True)
    parser.add_argument('--out',required=True,help='explicit dataset output folder')
    parser.add_argument('--force',action='store_true',help='rebuild this output after complete preparation')
    parser.add_argument('--dry-run',action='store_true',help='check the inputs without generating or writing data')
    parser.add_argument('--convert-from',help='copy an existing dataset to the new format without generation')
    parser.add_argument('--weights-dir',help='root of input model checkpoints')
    parser.add_argument('overrides',nargs='*')
    args=parser.parse_intermixed_args()
    with initialize_config_dir(config_dir=str(CFG_DIR),version_base=None):
        cfg=compose(config_name=args.config_name,overrides=args.overrides)
    result=prepare_dataset(cfg,args.out,force=args.force,dry_run=args.dry_run,
                           convert_from=args.convert_from,weights_dir=args.weights_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
