from ete3 import Tree
import numpy as np
import os

def read_forest(tree_path):
    with open(tree_path, 'r') as f:
        nwks = f.read().replace('\n', '').split(';')
    return [Tree(nwk + ';') for nwk in nwks[:-1]]



def remove_certain_leaves(tr, to_remove=lambda node: False):
    """
    Removes all the branches leading to leaves identified positively by to_remove function.
    :param tr: the tree of interest (ete3 Tree)
    :param to_remove: a method to check is a leaf should be removed.
    :return: void, modifies the initial tree.
    """

    tips = [tip for tip in tr if to_remove(tip)]
    for node in tips:
        if node.is_root():
            return None
        parent = node.up
        parent.remove_child(node)
        # If the parent node has only one child now, merge them.
        if len(parent.children) == 1:
            brother = parent.children[0]
            brother.dist += parent.dist
            if parent.is_root():
                brother.up = None
                tr = brother
            else:
                grandparent = parent.up
                grandparent.remove_child(parent)
                grandparent.add_child(brother)
    return tr



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chops extra tips from the tree")
    parser.add_argument('--in_nwk', default='/home/azhukova/mPath/anna/projects/bdext/sim_bdeiss/test/2000_5000/BDEI/tree.393.nwk', type=str, help="input tree")
    parser.add_argument('--out_nwk', default='/home/azhukova/mPath/anna/projects/bdext/sim_bdeiss/test/1000_2000/BDEI/tree.393.nwk', type=str, help="output tree")

    parser.add_argument('--min_tips', default=1000, type=int, help="min tips (included)")
    parser.add_argument('--max_tips', default=2000, type=int, help="max tips (included)")

    params = parser.parse_args()


    out_nwk = f'{params.out_nwk}.temp'
    with open(out_nwk, 'w+') as f:
        for tree in read_forest(params.in_nwk):
            n = np.random.randint(params.min_tips, params.max_tips + 1)
            if len(tree) > n:
                for node in tree.traverse('preorder'):
                    node.add_feature('date', 0 if node.is_root() else (getattr(node.up, 'date') + node.dist))
                tip_names = sorted(tree, key=lambda _: getattr(_, 'date'))
                tip_names = {_.name for _ in tip_names[n:]}
                print(f'Removing {len(tip_names)} tips from tree with {len(tree)} tips to get a tree with {n} tips.')
                tree = remove_certain_leaves(tree, to_remove=lambda tip: tip.name in tip_names)
            f.write(tree.write(format=5, format_root_node=True) + '\n')
    os.rename(out_nwk, params.out_nwk)
