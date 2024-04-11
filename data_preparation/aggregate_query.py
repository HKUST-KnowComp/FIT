import argparse
import json
import os.path as osp
import pandas as pd

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='valid', choices=['train', 'valid', 'test'])
parser.add_argument("--data_folder", type=str, default='data/FB15k-237-EFO1')
parser.add_argument("--output_folder", type=str, default='data/FB15k-237-EFO1')
parser.add_argument("--max", type=int, default=None)
parser.add_argument("--specific_max", type=bool, default=False)
parser.add_argument("--action", type=str, default='split', choices=['aggregate', 'split'])
parser.add_argument("--data_file", type=str, default="data/DNF_evaluate_EFO1.csv")
parser.add_argument("--data_type", type=str, default="EFO1", choices=['EFO1', 'EFO1ex'])
parser.add_argument("--output_prefix", type=str, default="real_EFO1")


formula_num_dict = {
    '((r1(s1,e1))&(!(r2(e1,f1))))&(r3(s2,f1))': 149689,
    '(r1(s1,e1))&((r2(e1,f1))&(r3(e1,f1)))': 149689,
    '(r1(s1,e1))&((r2(e1,f1))&(!(r3(e1,f1))))': 149689,
    '(r1(s1,e1))&((r2(s2,e1))&((r3(e1,f1))&(r4(e1,f1))))': 0,
    '(r1(s1,e1))&((r2(e1,e2))&((r3(e1,e2))&(r4(e2,f1))))': 0,
    '(r1(s1,e1))&((r2(e1,e2))&((r3(e2,f1))&(r4(e2,f1))))': 0,
    '(r1(s1,f1))&(r2(e1,f1))': 74845,
    '(r1(s1,f1))&((r2(s2,f1))&(r3(e1,f1)))': 74845,
    '(r1(s1,e1))&((r2(s2,e2))&((r3(e1,e2))&((r4(e1,f1))&(r5(e2,f1)))))': 0,
    '(r1(s1,e1))&((r2(s2,e2))&((r3(e1,f1))&((r4(e1,f1))&((r5(e1,e2))&(r6(e2,f1))))))': 0
}


pni_formula_dict = {
    '(r1(s1,e1))&((r2(e1,f1))&(!(r3(e1,f1))))': 14968
}


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    all_query_data = {}
    abstract_query_data = pd.read_csv(args.data_file)
    if args.action == 'aggregate':
        for formula_id in abstract_query_data['formula_id']:
            prefix = 'real_EFO1' if args.data_type == 'EFO1' else args.data_type
            query_path = osp.join(args.data_folder, f'{args.mode}_{formula_id}_{prefix}_qaa.json')
            if not osp.exists(query_path):
                print(f'File {query_path} does not exist.')
                continue
            query_file = open(query_path)
            query_data = json.load(query_file)
            for query in query_data:
                if args.max:
                    all_query_data[query] = query_data[query][:args.max]
                elif args.specific_max:
                    if query in formula_num_dict:
                        if formula_num_dict[query] != 0:
                            all_query_data[query] = query_data[query][:formula_num_dict[query]]
                    else:
                        all_query_data[query] = query_data[query]
                else:
                    all_query_data[query] = query_data[query]
        if args.max:
            output_path = osp.join(args.output_folder, f'{args.mode}_{args.max}_qaa.json')
        else:
            output_path = osp.join(args.output_folder, f'{args.mode}_qaa.json')
        with open(output_path, 'wt') as f:
            json.dump(all_query_data, f)
    else:
        if args.data_type == 'EFO1':
            query_file = open(osp.join(args.data_folder, f'{args.mode}_qaa.json'))
        elif args.data_type == 'EFO1ex':
            query_file = open(osp.join(args.data_folder, f'{args.mode}_real_EFO1_qaa.json'))
        else:
            raise NotImplementedError
        all_query_data = json.load(query_file)
        for query in all_query_data:
            if query in abstract_query_data['formula'].values:
                specific_query_data = {}
                formula_id = abstract_query_data['formula_id'][
                    abstract_query_data['formula'].index[abstract_query_data['formula'] == query][0]]
                if args.max:
                    specific_query_data[query] = all_query_data[query][:args.max]
                elif args.specific_max:
                    if query in pni_formula_dict:
                        if formula_num_dict[query] != 0:
                            specific_query_data[query] = all_query_data[query][:formula_num_dict[query]]
                    else:
                        continue
                else:
                    specific_query_data[query] = all_query_data[query]
                if args.max:
                    output_path = osp.join(args.output_folder,
                                           f'{args.mode}_{args.max}_{formula_id}_{args.data_type}_qaa.json')
                else:
                    output_path = osp.join(args.output_folder, f'{args.mode}_{formula_id}_{args.output_prefix}_qaa.json')
                with open(output_path, 'wt') as f:
                    json.dump(specific_query_data, f)



