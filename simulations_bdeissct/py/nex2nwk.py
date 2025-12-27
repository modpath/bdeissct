from datetime import datetime

from Bio import Phylo
import re

def read_nexus(tree_path):
    with open(tree_path, 'r') as f:
        nexus = f.read()
    # replace CI_date="2019(2018,2020)" with CI_date="2018 2020"
    nexus = re.sub(r'CI_date="({})\(({}),({})\)"'.format(DATE_REGEX, DATE_REGEX, DATE_REGEX), r'CI_date="\2 \3"',
                   nexus)
    temp = tree_path + '.{}.temp'.format(datetime.timestamp(datetime.now()))
    with open(temp, 'w') as f:
        f.write(nexus)
    trees = list(Phylo.parse(temp, 'nexus'))
    os.remove(temp)
    return trees

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a collection of nexus trees into newick.")
    parser.add_argument('--out_nwk', required=True, type=str, help="output nwk tree")
    parser.add_argument('--in_nexus', required=True, nargs='+', type=str, help="input nexus trees")

    params = parser.parse_args()
    output = [read_nexus(in_nexus)[0].write(format=5) for in_nexus in params.in_nexus]
    with open(params.out_nwk, 'w') as f:
        f.write('\n'.join(output))
