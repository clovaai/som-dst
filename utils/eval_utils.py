"""
Most codes are from https://github.com/jasonwu0731/trade-dst
"""

from .data_utils import EXPERIMENT_DOMAINS


def per_domain_join_accuracy(data, slot_temp):
    for dom in EXPERIMENT_DOMAINS:
        count = 0
        jt = 0
        acc = 0
        for k, d in data.items():
            p, g = d
            gg = [r for r in g if r.startswith(dom)]
            if len(gg) > 0:
                pp = [r for r in p if r.startswith(dom)]
                count += 1
                if set(pp) == set(gg):
                    jt += 1
                temp_acc = compute_acc(set(gg), set(pp), slot_temp)
                acc += temp_acc
        print(dom, jt / count, acc / count)


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count
