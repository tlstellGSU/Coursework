# Assignment 3 - Data mIning

# Stell

from itertools import combinations
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori as mlx_apriori
from mlxtend.frequent_patterns import association_rules, fpgrowth as mlx_fpgrowth
from mlxtend.preprocessing import TransactionEncoder


file_path = Path(__file__).resolve().parent
output_path = file_path / "HW_3_output"
output_path.mkdir(parents=True, exist_ok=True)


def fmt(itemset):
    # print an itemset as {A, B, C} in sorted order
    return "{" + ", ".join(sorted(itemset)) + "}"


# Shared implementations (written from scratch, used by Q2 and cross-checked in Q4)

def count_support(transactions, candidates):
    # one pass over the database: count how many transactions contain each candidate
    counts = {c: 0 for c in candidates}
    for t in transactions:
        for c in candidates:
            if c <= t:
                counts[c] += 1
    return counts


def apriori_own(transactions, min_count, verbose=False):
    transactions = [frozenset(t) for t in transactions]
    frequent = {}

    # scan 1: C1 and L1
    items = sorted({i for t in transactions for i in t})
    C = [frozenset([i]) for i in items]
    k = 1
    while C:
        counts = count_support(transactions, C)
        L = {c: n for c, n in counts.items() if n >= min_count}
        if verbose:
            print("\nC{} (scan {}): candidate counts".format(k, k))
            for c in sorted(C, key=sorted):
                print("  {:<22} {}{}".format(fmt(c), counts[c], "" if c in L else "   < min_count, dropped"))
            print("L{}: {}".format(k, ", ".join("{}:{}".format(fmt(c), n) for c, n in sorted(L.items(), key=lambda x: sorted(x[0])))
                                   if L else "(empty)"))
        frequent.update(L)

        # join step: merge pairs of frequent k-itemsets that share their first k-1 items
        Lk = sorted((tuple(sorted(c)) for c in L))
        joined = []
        for a, b in combinations(Lk, 2):
            if a[:-1] == b[:-1]:
                joined.append(frozenset(a) | frozenset(b))

        # prune step: drop any candidate with an infrequent k-subset
        C = []
        if verbose and joined:
            print("\nC{} from joining L{} with itself:".format(k + 1, k))
        for c in joined:
            bad = [frozenset(s) for s in combinations(sorted(c), k) if frozenset(s) not in L]
            if bad:
                if verbose:
                    print("  {:<22} pruned: subset {} not in L{}".format(fmt(c), ", ".join(fmt(b) for b in bad), k))
            else:
                if verbose:
                    print("  {:<22} kept (all {}-subsets frequent)".format(fmt(c), k))
                C.append(c)
        if verbose and not C:
            print("\nC{} is empty, so Apriori stops.".format(k + 1))
        k += 1
    return frequent


class FPNode:
    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}
        self.link = None


def build_fp_tree(weighted_transactions, min_count, rank=None):
    # weighted_transactions: list of (items, count). Scan 1 counts items, scan 2 inserts them.
    item_counts = {}
    for items, n in weighted_transactions:
        for i in items:
            item_counts[i] = item_counts.get(i, 0) + n
    # F-list: frequent items in descending count, ties broken alphabetically. A conditional
    # tree keeps the parent tree's F-list order (rank) so it reads as a piece of that tree.
    if rank is None:
        order = sorted((i for i, n in item_counts.items() if n >= min_count), key=lambda i: (-item_counts[i], i))
        rank = {i: r for r, i in enumerate(order)}
    else:
        order = sorted((i for i, n in item_counts.items() if n >= min_count), key=rank.get)
        rank = {i: rank[i] for i in order}

    root = FPNode(None, None)
    header = {i: None for i in order}
    sorted_transactions = []
    for items, n in weighted_transactions:
        path = sorted((i for i in items if i in rank), key=rank.get)
        sorted_transactions.append((path, n))
        node = root
        for i in path:
            if i not in node.children:
                child = FPNode(i, node)
                node.children[i] = child
                # thread the new node onto the header table's node-link chain
                child.link = header[i]
                header[i] = child
            node = node.children[i]
            node.count += n
    return root, header, order, {i: item_counts[i] for i in order}, sorted_transactions


def print_tree(node, depth=0):
    for child in node.children.values():
        print("  " + "    " * depth + "{}:{}".format(child.item, child.count))
        print_tree(child, depth + 1)


def fp_growth_own(transactions, min_count, verbose=False, labels=None):
    root, header, order, counts, sorted_transactions = build_fp_tree([(t, 1) for t in transactions], min_count)
    if verbose:
        print("\nScan 1, header table (descending count, ties broken alphabetically):")
        for i in order:
            print("  {}: {}".format(i, counts[i]))
        print("F-list order: " + " > ".join(order))
        print("\nScan 2, each transaction's frequent items in F-list order:")
        for tid, (path, _) in zip(labels or range(len(sorted_transactions)), sorted_transactions):
            print("  {}: {}".format(tid, ", ".join(path)))
        print("\nFP-tree (item:count, indentation = depth below the null root):")
        print_tree(root)
    frequent = {}
    mine(header, order, counts, frozenset(), min_count, frequent, verbose, {i: r for r, i in enumerate(order)})
    return frequent


def mine(header, order, counts, suffix, min_count, frequent, verbose, rank):
    # process items bottom-up (least frequent first)
    for item in reversed(order):
        pattern = suffix | {item}
        frequent[pattern] = counts[item]

        # conditional pattern base: the prefix path above every node of this item
        base = []
        node = header[item]
        while node is not None:
            path = []
            p = node.parent
            while p.item is not None:
                path.append(p.item)
                p = p.parent
            if path:
                base.append((list(reversed(path)), node.count))
            node = node.link

        sub_root, sub_header, sub_order, sub_counts, _ = build_fp_tree(base, min_count, rank)
        if verbose:
            print("\nItem {} (suffix {}):".format(item, fmt(pattern)))
            # each prefix path is printed root-to-node, in F-list order
            print("  conditional pattern base: " +
                  ("; ".join("<{}>:{}".format(", ".join(p), n) for p, n in base) if base else "(empty)"))
            if sub_order:
                print("  conditional FP-tree:")
                print_tree(sub_root, 1)
            else:
                print("  conditional FP-tree: empty (no item reaches min_count)")
        mine(sub_header, sub_order, sub_counts, pattern, min_count, frequent, False, rank)
        if verbose:
            found = sorted((p for p in frequent if item in p and p <= pattern | set(sub_order)), key=lambda p: (len(p), sorted(p)))
            print("  frequent patterns ending in {}: {}".format(
                item, ", ".join("{}:{}".format(fmt(p), frequent[p]) for p in found if p >= pattern)))
    return frequent


def print_itemsets(frequent, n_transactions):
    for p, n in sorted(frequent.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
        print("  {:<22} count {}   support {:.0%}".format(fmt(p), n, n / n_transactions))


# Question 2

Q2_dict = {
    'T100': ['M', 'O', 'N', 'K', 'E', 'Y'],
    'T200': ['D', 'O', 'N', 'K', 'E', 'Y'],
    'T300': ['M', 'A', 'K', 'E'],
    'T400': ['M', 'U', 'C', 'K', 'Y'],
    'T500': ['C', 'O', 'K', 'I', 'E']
    }

Q2_min_sup = 0.60
Q2_min_conf = 0.80


def Q2_main():
    transactions = [set(v) for v in Q2_dict.values()]
    n = len(transactions)
    # an itemset is frequent if its count >= 60% of 5 = 3 transactions
    min_count = Q2_min_sup * n
    print("Q2: {} transactions, min_sup = {:.0%} -> min_count = {:g}".format(n, Q2_min_sup, min_count))

    # Q2a
    print("\n===== Q2a: Apriori =====")
    apriori_result = apriori_own(transactions, min_count, verbose=True)
    print("\nAll frequent itemsets (Apriori):")
    print_itemsets(apriori_result, n)

    # Q2b
    print("\n===== Q2b: FP-growth =====")
    fp_result = fp_growth_own(transactions, min_count, verbose=True, labels=list(Q2_dict))
    print("\nAll frequent itemsets (FP-growth):")
    print_itemsets(fp_result, n)
    assert fp_result == apriori_result, "FP-growth and Apriori disagree"
    print("\nCheck passed: FP-growth and Apriori return the same itemsets and counts.")

    # Q2d
    print("\n===== Q2d: rules buys(X, item1) and buys(X, item2) => buys(X, item3) =====")
    print("min_sup = {:.0%}, min_conf = {:.0%}".format(Q2_min_sup, Q2_min_conf))
    for itemset, count in apriori_result.items():
        if len(itemset) != 3:
            continue
        for consequent in sorted(itemset):
            antecedent = itemset - {consequent}
            support = count / n
            confidence = count / apriori_result[antecedent]
            strong = support >= Q2_min_sup and confidence >= Q2_min_conf
            print("  {} => {{{}}}   support = {}/{} = {:.0%}   confidence = {}/{} = {:.0%}   {}".format(
                fmt(antecedent), consequent, count, n, support, count, apriori_result[antecedent], confidence,
                "STRONG" if strong else "not strong (confidence < {:.0%})".format(Q2_min_conf)))


# Question 4

Q4_dataset_path = file_path / "retail_200_expanded.csv"
Q4_min_sup = 0.25
Q4_min_conf = 0.80


def Q4_main():
    data = pd.read_csv(Q4_dataset_path)
    # items are comma separated and some contain spaces ("Whole Milk"), so split on commas and strip
    transactions = [[i.strip() for i in row.split(",")] for row in data['Items']]
    n = len(transactions)
    encoder = TransactionEncoder()
    onehot = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)
    print("\n===== Q4: {} transactions, {} distinct items =====".format(n, len(encoder.columns_)))
    print("min_sup = {:.0%} -> an itemset needs at least {:g} transactions".format(Q4_min_sup, Q4_min_sup * n))

    # Frequent itemsets a) and b)
    freq_apriori = mlx_apriori(onehot, min_support=Q4_min_sup, use_colnames=True)
    freq_fpgrowth = mlx_fpgrowth(onehot, min_support=Q4_min_sup, use_colnames=True)

    # c) compare as {itemset: support} so row order does not matter
    a = {s: round(sup * n) for s, sup in zip(freq_apriori['itemsets'], freq_apriori['support'])}
    f = {s: round(sup * n) for s, sup in zip(freq_fpgrowth['itemsets'], freq_fpgrowth['support'])}
    assert a == f, "Apriori and FP-Growth returned different frequent itemsets"
    print("Check passed: mlxtend Apriori and FP-Growth return the same {} frequent itemsets with the same counts.".format(len(a)))

    # cross-check against the from-scratch implementations used in Q2
    assert apriori_own(transactions, Q4_min_sup * n) == a, "own Apriori disagrees with mlxtend"
    assert fp_growth_own(transactions, Q4_min_sup * n) == a, "own FP-growth disagrees with mlxtend"
    print("Check passed: the Q2 from-scratch Apriori and FP-growth give the same result on this data.")

    # every itemset with count >= 50 is already in a, so any with exactly 50 sit on the boundary
    boundary = [fmt(s) for s, c in a.items() if c == Q4_min_sup * n]
    print("Itemsets with a count of exactly {:g} (kept by >= 25%, dropped by > 25%): {}".format(
        Q4_min_sup * n, ", ".join(boundary) if boundary else "none, so >= and > give the same result"))

    print("\n1-itemsets:")
    for s, c in sorted(((s, c) for s, c in a.items() if len(s) == 1), key=lambda x: -x[1]):
        print("  {:<30} {:>4}   {:.1%}".format(fmt(s), c, c / n))
    table = pd.DataFrame([(fmt(s), len(s), c, c / n) for s, c in a.items() if len(s) >= 2],
                         columns=['itemset', 'length', 'frequency', 'support'])
    table = table.sort_values(['length', 'frequency'], ascending=[True, False]).reset_index(drop=True)
    print("\nFrequent itemsets of length >= 2 (frequency = number of transactions):")
    print(table.to_string(formatters={'support': '{:.1%}'.format}))
    table.to_csv(output_path / "Q4_frequent_itemsets.csv", index=False)

    # Rule resolution
    rules = association_rules(freq_fpgrowth, num_itemsets=n, metric='confidence', min_threshold=Q4_min_conf)
    # check every rule with integer counts, independently of mlxtend: keep A => L - A when
    # count(L)/count(A) >= 0.8, i.e. 5 count(L) >= 4 count(A), so a confidence of exactly
    # 0.8 (e.g. Bread => Eggs, 120/150) cannot be lost to floating-point rounding
    own_rules = {}
    for L, cL in a.items():
        for r in range(1, len(L)):
            for A in map(frozenset, combinations(sorted(L), r)):
                if 5 * cL >= 4 * a[A]:
                    own_rules[(A, L - A)] = cL * n / (a[A] * a[L - A])
    mlx_rules = {(x, y): l for x, y, l in zip(rules['antecedents'], rules['consequents'], rules['lift'])}
    assert own_rules.keys() == mlx_rules.keys(), "rule sets differ from the integer-count check"
    assert all(abs(own_rules[k] - mlx_rules[k]) < 1e-9 for k in own_rules), "lift differs from the integer-count check"
    print("\nCheck passed: all {} mlxtend rules (and their lifts) match an integer-count rule generator.".format(len(own_rules)))

    # sensitivity: what a strictly-greater-than 25% threshold would remove
    strict = {s: c for s, c in a.items() if c > Q4_min_sup * n}
    strict_rules = [k for k in own_rules if (k[0] | k[1]) in strict]
    print("Under support > 25% instead of >= 25%: {} of {} itemsets of length >= 2 and {} of {} rules remain.".format(
        sum(len(s) >= 2 for s in strict), sum(len(s) >= 2 for s in a), len(strict_rules), len(own_rules)))

    rules['rule'] = [fmt(x) + " => " + fmt(y) for x, y in zip(rules['antecedents'], rules['consequents'])]
    # exact ties in lift (e.g. Bread => Eggs and Eggs => Bread, both 8/7) can differ in the last
    # floating-point bit, so sort on a rounded copy and let the tie-break keys decide the order
    rules['lift_key'] = rules['lift'].round(10)
    rules = rules.sort_values(['lift_key', 'confidence', 'support', 'rule'],
                              ascending=[False, False, False, True]).reset_index(drop=True)
    out = pd.DataFrame({
        'rule': rules['rule'],
        'support': rules['support'],
        'confidence': rules['confidence'],
        'consequent support': rules['consequent support'],
        'lift': rules['lift'],
    })
    print("\nRules with confidence >= {:.0%}, sorted by lift (descending):".format(Q4_min_conf))
    print(out.to_string(formatters={'support': '{:.3f}'.format, 'confidence': '{:.3f}'.format,
                                    'consequent support': '{:.3f}'.format, 'lift': '{:.3f}'.format}))
    out.to_csv(output_path / "Q4_rules.csv", index=False)

    weak = out[out['lift'] <= 1]
    print("\n{} of {} rules have lift > 1 (positively correlated); {} have lift <= 1.".format(
        len(out) - len(weak), len(out), len(weak)))
    if len(weak):
        print("Not meaningful by lift (consequent is at least as likely without the antecedent):")
        print(weak.to_string(formatters={'support': '{:.3f}'.format, 'confidence': '{:.3f}'.format,
                                         'consequent support': '{:.3f}'.format, 'lift': '{:.3f}'.format}))

    near_one = out[(out['lift'] > 1) & (out['lift'] < 1.2)]
    print("\n{} rules have 1 < lift < 1.2, i.e. confidence only slightly above the consequent's base rate:".format(len(near_one)))
    print(near_one.to_string(formatters={'support': '{:.3f}'.format, 'confidence': '{:.3f}'.format,
                                         'consequent support': '{:.3f}'.format, 'lift': '{:.3f}'.format}))


# run

if __name__ == "__main__":
    Q2_main()
    Q4_main()
