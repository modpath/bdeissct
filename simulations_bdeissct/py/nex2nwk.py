from datetime import datetime

from Bio import Phylo
from ete3 import TreeNode
import os
import re
import numpy as np

ONE_HOUR = 0.0416951
DATE_REGEX = r'[+-]*[\d]+[.\d]*(?:[e][+-][\d]+){0,1}'

def read_nexus(tree_path):
    with open(tree_path, 'r') as f:
        nexus = f.read()
    # replace CI_date="2019(2018,2020)" with CI_date="2018 2020"
    nexus = re.sub(r'CI_date="({})\(({}),({})\)"'.format(DATE_REGEX, DATE_REGEX, DATE_REGEX), r'CI_date="\2 \3"',
                   nexus)
    temp = tree_path + '.{}.temp'.format(datetime.timestamp(datetime.now()))
    with open(temp, 'w') as f:
        f.write(nexus)
    nex_trees = list(Phylo.parse(temp, 'nexus'))
    os.remove(temp)

    trees = []
    for nex_tree in nex_trees:
        todo = [(nex_tree.root, None)]
        tree = None
        while todo:
            clade, parent = todo.pop()
            dist = 0
            try:
                dist = float(clade.branch_length)
            except:
                pass
            name = getattr(clade, 'name', None)
            if not name:
                name = getattr(clade, 'confidence', None)
                if not isinstance(name, str):
                    name = None
            node = TreeNode(dist=dist, name=name)
            if parent is None:
                tree = node
            else:
                parent.add_child(node)
            todo.extend((c, node) for c in clade.clades)
        trees.append(tree)
    return trees


def resolve_polytomies(tree, zero_dist=ONE_HOUR):
    """
    Resolves polytomies in the input tree by adding branches
    of a given length.

    :param zero_dist: float, length of the branches to be added to resolve polytomies
    :param tree: ete3.Tree, tree to be modified
    :return: ete3.Tree, the modified original tree
    """

    todo = [tree]
    while todo:
        n = todo.pop()
        children = list(n.children)
        if len(children) > 2:
            np.random.shuffle(children)
            threshold = np.random.randint(1, len(children) - 1)
            if threshold > 1:
                left_child = TreeNode(dist=zero_dist)
                for child in children[:threshold]:
                    n.remove_child(child)
                    left_child.add_child(child)
                n.add_child(left_child)
            if threshold < len(children) - 1:
                right_child = TreeNode(dist=zero_dist)
                for child in children[threshold:]:
                    n.remove_child(child)
                    right_child.add_child(child)
                n.add_child(right_child)
        if n.is_leaf() and n.dist == 0:
            n.dist = zero_dist
        todo.extend(n.children)
    return tree




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a collection of nexus trees into newick.")
    parser.add_argument('--out_nwk', type=str, help="output nwk tree")
    parser.add_argument('--in_nexus', nargs='+', type=str, help="input nexus trees")
    parser.add_argument('--min_brlen', default=ONE_HOUR, type=float, help="minimal branch length to set")

    params = parser.parse_args()
    with open(params.out_nwk, 'w') as f:
        for in_nexus in params.in_nexus:
            f.write(resolve_polytomies(read_nexus(in_nexus)[0]).write(format=5))
            f.write('\n')
