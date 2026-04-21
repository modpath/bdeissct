import logging
import os

import numpy as np
from bdeissct_dl.tree_manager import read_forest, remove_certain_leaves


def chop_main(in_nwk, out_nwk, min_tips, max_tips, verbose=False):

    logging.getLogger().setLevel(level=logging.INFO if verbose else logging.ERROR)

    out_nwk_temp = f'{out_nwk}.temp'
    with open(out_nwk_temp, 'w+') as f:
        for tree in read_forest(in_nwk):
            if len(tree) < min_tips:
                raise ValueError(
                    f'The tree has {len(tree)} tips, which is less than the minimum number of tips ({min_tips}).')
            n = np.random.randint(min_tips, max_tips + 1)
            if len(tree) > n:
                for node in tree.traverse('preorder'):
                    node.add_feature('date', 0 if node.is_root() else (getattr(node.up, 'date') + node.dist))
                tip_names = sorted(tree, key=lambda _: getattr(_, 'date'))
                tip_names = {_.name for _ in tip_names[n:]}
                logging.info(
                    f'Removing {len(tip_names)} tips from tree with {len(tree)} tips to get a tree with {n} tips.')
                tree = remove_certain_leaves(tree, to_remove=lambda tip: tip.name in tip_names)
            f.write(tree.write(format=5, format_root_node=True) + '\n')
    os.rename(out_nwk_temp, out_nwk)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chops extra (most recent) tips from the tree to obtain a tree of a given size.")
    parser.add_argument('--in_nwk', type=str, help="input tree")
    parser.add_argument('--out_nwk', type=str, help="output tree")

    parser.add_argument('--min_tips', type=int, help="min tips (included)")
    parser.add_argument('--max_tips', type=int, help="max tips (included)")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="whether to print information about the generated parameters and trees")
    params = parser.parse_args()

    chop_main(**vars(params))



if '__main__' == __name__:
    main()