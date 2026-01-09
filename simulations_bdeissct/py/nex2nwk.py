from datetime import datetime

from Bio import Phylo
from ete3 import TreeNode
import os
import re
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

nex = [f'/home/azhukova/mPath/anna/projects/bdext/sim_bdeiss/covid_train/200_500/BD/12/tree.79.{i}.nexus' for i in range(8)]
nwk = '/home/azhukova/mPath/anna/projects/bdext/sim_bdeiss/covid_train/200_500/BD/12/trees.79.nwk'

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a collection of nexus trees into newick.")
    parser.add_argument('--out_nwk', default=nwk, type=str, help="output nwk tree")
    parser.add_argument('--in_nexus', default=nex, nargs='+', type=str, help="input nexus trees")

    params = parser.parse_args()
    output = [read_nexus(in_nexus)[0].write(format=5) for in_nexus in params.in_nexus]
    with open(params.out_nwk, 'w') as f:
        f.write('\n'.join(output))
