from datetime import datetime
from ete3 import Tree, TreeNode

import numpy as np
from collections import Counter

MONTH = 'month'

DAY = 'day'

YEAR = 'year'

def read_forest(tree_path):
    with open(tree_path, 'r') as f:
        nwks = f.read().replace('\n', '').split(';')
    for nwk in nwks[:-1]:
        yield Tree(nwk + ';')


def datetime2numeric(d):
    """
    Converts a datetime date to numeric format.
    For example: 2016-12-31 -> 2016.9972677595629; 2016-1-1 -> 2016.0
    :param d: a date to be converted
    :type d: np.datetime
    :return: numeric representation of the date
    :rtype: float
    """
    first_jan_this_year = datetime(year=d.year, month=1, day=1)
    day_of_this_year = d - first_jan_this_year
    first_jan_next_year = datetime(year=d.year + 1, month=1, day=1)
    days_in_this_year = first_jan_next_year - first_jan_this_year
    return d.year + day_of_this_year / days_in_this_year



def numeric2datetime(d):
    """
    Converts a numeric date to  datetime format.
    For example: 2016.9972677595629 -> 2016-12-31; 2016.0 ->  2016-1-1
    :param d: numeric representation of a date to be converted
    :type d: float
    :return: the converted date
    :rtype: np.datetime
    """
    year = int(d)
    first_jan_this_year = datetime(year=year, month=1, day=1)
    first_jan_next_year = datetime(year=year + 1, month=1, day=1)
    days_in_this_year = first_jan_next_year - first_jan_this_year
    day_of_this_year = int(round(days_in_this_year.days * (d % 1), 6)) + 1
    for m in range(1, 13):
        days_in_m = (datetime(year=year if m < 12 else (year + 1), month=m % 12 + 1, day=1)
                     - datetime(year=year, month=m, day=1)).days
        if days_in_m >= day_of_this_year:
            return datetime(year=year, month=m, day=day_of_this_year)
        day_of_this_year -= days_in_m


def discretize_branch_lengths(tree, R_mean=0.0008, R_std=0.0004, s=29903):
    """
    Timetree-to-genetic tree converter.

    The method:
    1. Converts the rate into [substitutions per time unit]: r = R s, where R is the clock rate and s is the sequence length
    2. Converts every branch of  the timetree to distance [substitutions per site] by:
         2a. Taking the branch length t measured in time
         2b. Drawing a number of substitutions m along it from a Poisson distribution (with a rate r over time t)
         2c. Dividing m by the alignment length s: m / s [substitutions per site]
    3. Saves it as a newick file

    :param tree: ete3.Tree, tree to be modified
    :param s: int, number of sites in the alignment
    :param R_mean: float, mean clock rate [substitutions/site/time unit]
    :param R_std: float, standard deviation of clock rate
    timeunit: str, time unit of the input tree (year, month, day)
    :return: void, the original tree is modified
    """
    # Convert to lognormal parameters (mu and sigma)
    sigma_squared = np.log(1 + (R_std / R_mean) ** 2)
    sigma = np.sqrt(sigma_squared)
    mu = np.log(R_mean) - sigma_squared / 2

    for node in tree.traverse('preorder'):
        # distance in time
        t = node.dist
        if t > 0:
            r = np.random.lognormal(mean=mu, sigma=sigma) * s # substitutions per time unit
            # print(f'Expected {r} substitutions per time unit')
            subs = np.random.poisson(r * t)
            # print(f'Got {subs} substitutions on a branch of length {node.dist} timeunits')
            node.dist = subs / s # substitutions per site
    return None

def extract_dates(tree, root_date_min=10, root_date_max=12):
    """
    Extract tip dates (truncated to integers) from a transmission tree.

    :param tree: ete3.Tree, transmission tree
    :param root_date_min: float, minimum root date (inclusive)
    :param root_date_max: float, maximum root date (exclusive)

    :return: dict, mapping tip names to their dates
    """
    name2date = {}
    root_date = np.random.uniform(root_date_min, root_date_max)
    for node in tree.traverse('preorder'):
        # distance in time
        t = node.dist
        node.add_feature('date', root_date if node.is_root() else (getattr(node.up, 'date') + t))
        if node.is_leaf():
            # truncate the date
            name2date[node.name] = int(node.date)
    return name2date

def collapse_zero_branches(tree):
    """
    Collapses zero branches in the input tree.

    :param tree: ete3.Tree, tree to be collapsed
    :return: void, the original tree is modified
    """
    num_collapsed, num_total = 0, 0
    for n in list(tree.traverse('postorder')):
        num_total += len(n.children)
        zero_children = [child for child in n.children if not child.is_leaf() and child.dist <= 0]
        if not zero_children:
            continue
        for child in zero_children:
            n.remove_child(child)
            for grandchild in child.children:
                n.add_child(grandchild)
        num_collapsed += len(zero_children)
    if num_collapsed:
        print('Collapsed {} zero branches out of {} branches.'.format(num_collapsed, num_total))



def name_tree(tree):
    """
    Names all the tree nodes that are not named or have non-unique names, with unique names.

    :param tree: tree to be named
    :type tree: ete3.Tree

    :return: void, modifies the original tree
    """
    existing_names = Counter()
    n_nodes = 0
    for _ in tree.traverse():
        n_nodes += 1
        if _.name:
            existing_names[_.name] += 1
    if n_nodes == len(existing_names):
        return
    i = 0
    new_existing_names = Counter()
    for node in tree.traverse('preorder'):
        name_prefix = 'root' if node.is_root() else ('t' if node.is_leaf() else 'n')
        name = 'root' if node.is_root() else node.name
        while name is None or name in new_existing_names:
            name = '{}{}'.format(name_prefix, i)
            i += 1
        node.name = name
        new_existing_names[name] += 1




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Converts a transmission tree into a date file + a phylogeny.")
    parser.add_argument('--out_dates', type=str, help="pattern for output date files")
    parser.add_argument('--in_nwk', type=str, help="input transmission trees")
    parser.add_argument('--out_nwk', type=str, help="pattern for output phylogenetic trees")

    parser.add_argument('--mean_R', default=0.0008, type=float, help="mean substitution rate [substitutions/site/year]")
    parser.add_argument('--std_R', default=0.0004, type=float, help="std of the substitution rate")
    parser.add_argument('--min_date', default=10, type=float, help="min root date [in tree time units] (included)")
    parser.add_argument('--max_date', default=12, type=float, help="max root datee [in tree time units] (excluded)")
    parser.add_argument('--time_scale', default=DAY, choices=(DAY, MONTH, YEAR), help="time unit of the input tree")
    parser.add_argument('--s', default=29903, type=int, help="alignment length (number of sites)")

    params = parser.parse_args()
    for i, tree in enumerate(read_forest(params.in_nwk)):
        name_tree(tree)
        scale = 365 if params.time_scale == DAY else 12 if timeunit == MONTH else 1

        name2date = extract_dates(tree, root_date_min=params.min_date, root_date_max=params.max_date)
        discretize_branch_lengths(tree, R_mean=params.mean_R / scale, R_std=params.std_R / scale, s=params.s)
        collapse_zero_branches(tree)

        # resolve root
        if len(tree.children) > 2:
            children = list(tree.children)
            right_child = TreeNode(dist=0)
            for child in children[1:]:
                tree.remove_child(child)
                right_child.add_child(child)
            tree.add_child(right_child)

        # add minimal distances to tips
        for tip in tree:
            if tip.dist <= 0:
                tip.dist = 0.5  / params.s

        tree.write(format=3, outfile=params.out_nwk.replace('*', f'{i}'), format_root_node=True)
        with open(params.out_dates.replace('*', f'{i}'), 'w+') as f:
            f.write(f'{len(name2date) + 1}\n')
            left_tip = next(_.name for _ in tree.children[0])
            right_tip = next(_.name for _ in tree.children[1])
            f.write(f'mrca({left_tip},{right_tip})\tb({params.min_date},{params.max_date})\n')
            for name, date in name2date.items():
                f.write(f'{name}\t{date}\n')
