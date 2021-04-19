"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import numpy as np
import json
from torch.utils.data import Dataset
import torch
import random
import re
from copy import deepcopy
from .fix_label import fix_general_label_error

flatten = lambda x: [i for s in x for i in s]
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}


def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4', dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    op_labels = ['carryover'] * len(slot_meta)
    generate_y = []
    keys = list(turn_dialog_state.keys())
    for k in keys:
        v = turn_dialog_state[k]
        if v == 'none':
            turn_dialog_state.pop(k)
            continue
        vv = last_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv != v:
                if v == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
                    op_labels[idx] = 'dontcare'
                elif v == 'yes' and OP_SET[op_code].get('yes') is not None:
                    op_labels[idx] = 'yes'
                elif v == 'no' and OP_SET[op_code].get('no') is not None:
                    op_labels[idx] = 'no'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([tokenizer.tokenize(v) + ['[EOS]'], idx])
            elif vv == v:
                op_labels[idx] = 'carryover'
        except ValueError:
            continue

    for k, v in last_dialog_state.items():
        vv = turn_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv is None:
                if OP_SET[op_code].get('delete') is not None:
                    op_labels[idx] = 'delete'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([['[NULL]', '[EOS]'], idx])
        except ValueError:
            continue
    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y, gold_state


def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, tokenizer, op_code, gold_gen={}):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
            last_dialog_state[st] = 'dontcare'
        elif op == 'yes' and OP_SET[op_code].get('yes') is not None:
            last_dialog_state[st] = 'yes'
        elif op == 'no' and OP_SET[op_code].get('no') is not None:
            last_dialog_state[st] = 'no'
        elif op == 'delete' and last_dialog_state.get(st) and OP_SET[op_code].get('delete') is not None:
            last_dialog_state.pop(st)
        elif op == 'update':
            g = tokenizer.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == '[EOS]':
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '')
            if gold_gen and gold_gen.get(st) and gold_gen[st] not in ['dontcare']:
                gen = gold_gen[st]

            if gen == '[NULL]' and last_dialog_state.get(st) and not OP_SET[op_code].get('delete') is not None:
                last_dialog_state.pop(st)
            else:
                last_dialog_state[st] = gen
    return generated, last_dialog_state


def make_slot_meta(ontology):
    meta = []
    change = {}
    idx = 0
    max_len = 0
    for i, k in enumerate(ontology.keys()):
        d, s = k.split('-')
        if d not in EXPERIMENT_DOMAINS:
            continue
        if 'price' in s or 'leave' in s or 'arrive' in s:
            s = s.replace(' ', '')
        ss = s.split()
        if len(ss) + 1 > max_len:
            max_len = len(ss) + 1
        meta.append('-'.join([d, s]))
        change[meta[-1]] = ontology[k]
    return sorted(meta), change


def prepare_dataset(data_path, tokenizer, slot_meta,
                    n_history, max_seq_length, diag_level=False, op_code='4'):
    dials = json.load(open(data_path))
    data = []
    domain_counter = {}
    max_resp_len, max_value_len = 0, 0
    max_line = None
    for dial_dict in dials:
        for domain in dial_dict["domains"]:
            if domain not in EXPERIMENT_DOMAINS:
                continue
            if domain not in domain_counter.keys():
                domain_counter[domain] = 0
            domain_counter[domain] += 1

        dialog_history = []
        previous_dialog_state = {}  # previous dialog_state
        last_uttr = ""
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_domain = turn["domain"]
            if turn_domain not in EXPERIMENT_DOMAINS:
                continue
            turn_id = turn["turn_idx"]
            turn_uttr = (turn["system_transcript"] + ' ; ' + turn["transcript"]).strip()
            dialog_history.append(last_uttr)
            turn_dialog_state = fix_general_label_error(turn["belief_state"], False, slot_meta)
            last_uttr = turn_uttr

            op_labels, generate_y, gold_state = make_turn_label(slot_meta, previous_dialog_state,
                                                                turn_dialog_state,
                                                                tokenizer, op_code)
            if (ti + 1) == len(dial_dict["dialogue"]):
                is_last_turn = True
            else:
                is_last_turn = False

            instance = TrainingInstance(dial_dict["dialogue_idx"], turn_domain,
                                        turn_id, turn_uttr, ' '.join(dialog_history[-n_history:]),
                                        previous_dialog_state, op_labels,
                                        generate_y, gold_state, max_seq_length, slot_meta,
                                        is_last_turn, op_code=op_code)
            instance.make_instance(tokenizer)
            data.append(instance)
            previous_dialog_state = turn_dialog_state
    return data


class TrainingInstance:
    def __init__(self, ID,
                 turn_domain,
                 turn_id,
                 turn_utter,
                 dialog_history,
                 last_dialog_state,
                 op_labels,
                 generate_y,
                 gold_state,
                 max_seq_length,
                 slot_meta,
                 is_last_turn,
                 op_code='4'):
        self.id = ID
        self.turn_domain = turn_domain
        self.turn_id = turn_id
        self.turn_utter = turn_utter
        self.dialog_history = dialog_history
        self.last_dialog_state = last_dialog_state
        self.gold_p_state = last_dialog_state
        self.generate_y = generate_y
        self.op_labels = op_labels
        self.gold_state = gold_state
        self.max_seq_length = max_seq_length
        self.slot_meta = slot_meta
        self.is_last_turn = is_last_turn
        self.op2id = OP_SET[op_code]

    def shuffle_state(self, rng, slot_meta=None):
        new_y = []
        gid = 0
        for idx, aa in enumerate(self.op_labels):
            if aa == 'update':
                new_y.append(self.generate_y[gid])
                gid += 1
            else:
                new_y.append(["dummy"])
        if slot_meta is None:
            temp = list(zip(self.op_labels, self.slot_meta, new_y))
            rng.shuffle(temp)
        else:
            indices = list(range(len(slot_meta)))
            for idx, st in enumerate(slot_meta):
                indices[self.slot_meta.index(st)] = idx
            temp = list(zip(self.op_labels, self.slot_meta, new_y, indices))
            temp = sorted(temp, key=lambda x: x[-1])
        temp = list(zip(*temp))
        self.op_labels = list(temp[0])
        self.slot_meta = list(temp[1])
        self.generate_y = [yy for yy in temp[2] if yy != ["dummy"]]

    def make_instance(self, tokenizer, max_seq_length=None,
                      word_dropout=0., slot_token='[SLOT]'):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        state = []
        for s in self.slot_meta:
            state.append(slot_token)
            k = s.split('-')
            v = self.last_dialog_state.get(s)
            if v is not None:
                k.extend(['-', v])
                t = tokenizer.tokenize(' '.join(k))
            else:
                t = tokenizer.tokenize(' '.join(k))
                t.extend(['-', '[NULL]'])
            state.extend(t)
        avail_length_1 = max_seq_length - len(state) - 3    # 1 * CLS + 2 * SEP
        diag_1 = tokenizer.tokenize(self.dialog_history)
        diag_2 = tokenizer.tokenize(self.turn_utter)
        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncates diag_1 (dialog_history)
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:   # truncates diag_2 (turn_utter)
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = ["[CLS]"] + diag_1 + ["[SEP]"]
        diag_2 = diag_2 + ["[SEP]"]
        segment = [0] * len(diag_1) + [1] * len(diag_2)

        diag = diag_1 + diag_2
        # word dropout
        if word_dropout > 0.:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype('int64'), word_dropout)
            diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]

        input_ = diag + state
        segment = segment + [1]*len(state)
        self.input_ = input_

        self.segment_id = segment
        slot_position = []
        for i, t in enumerate(self.input_):
            if t == slot_token:     # delimiter "[SLOT]"
                slot_position.append(i)
        self.slot_position = slot_position

        input_mask = [1] * len(self.input_)
        self.input_id = tokenizer.convert_tokens_to_ids(self.input_)
        if len(input_mask) < max_seq_length:
            self.input_id = self.input_id + [0] * (max_seq_length-len(input_mask))

            # 000 (history), 11111 (turn_utter), 00000 (mask)
            self.segment_id = self.segment_id + [0] * (max_seq_length-len(input_mask))
            input_mask = input_mask + [0] * (max_seq_length-len(input_mask))

        self.input_mask = input_mask
        self.domain_id = domain2id[self.turn_domain]
        self.op_ids = [self.op2id[a] for a in self.op_labels]
        self.generate_ids = [tokenizer.convert_tokens_to_ids(y) for y in self.generate_y]


class MultiWozDataset(Dataset):
    def __init__(self, data, tokenizer, slot_meta, max_seq_length, rng,
                 ontology, word_dropout=0.1, shuffle_state=False, shuffle_p=0.5):
        self.data = data
        self.len = len(data)
        self.tokenizer = tokenizer
        self.slot_meta = slot_meta
        self.max_seq_length = max_seq_length
        self.ontology = ontology
        self.word_dropout = word_dropout
        self.shuffle_state = shuffle_state
        self.shuffle_p = shuffle_p
        self.rng = rng

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.shuffle_state and self.shuffle_p > 0.:
            if self.rng.random() < self.shuffle_p:
                self.data[idx].shuffle_state(self.rng, None)
            else:
                self.data[idx].shuffle_state(self.rng, self.slot_meta)
        if self.word_dropout > 0 or self.shuffle_state:
            self.data[idx].make_instance(self.tokenizer,
                                         word_dropout=self.word_dropout)
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        state_position_ids = torch.tensor([f.slot_position for f in batch], dtype=torch.long)
        op_ids = torch.tensor([f.op_ids for f in batch], dtype=torch.long)
        domain_ids = torch.tensor([f.domain_id for f in batch], dtype=torch.long)
        gen_ids = [b.generate_ids for b in batch]
        max_update = max([len(b) for b in gen_ids])
        max_value = max([len(b) for b in flatten(gen_ids)])
        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [0] * (max_value - len(v))
            gen_ids[bid] = b + [[0] * max_value] * (max_update - n_update)
        gen_ids = torch.tensor(gen_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids, state_position_ids, op_ids, domain_ids, gen_ids, max_value, max_update
