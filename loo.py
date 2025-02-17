import argparse
import copy
import os.path
import pandas as pd

from utils.data_utils import read_multi_row_data, read_multi_attr_data, read_single_row_data
from utils.train_eval import train, inference
from data import T5Dataset, GPTDataset, BertDataset, LlamaDataset
from model import load_model
products=['computers','shoes','watches','cameras']

def get_loo_dirs(dataset_name):
    dataset_names = ['abt', 'amgo', 'beer', 'dbac', 'dbgo', 'foza', 'itam', 'waam', 'wdc','music']
    dataset_names = [f'wdc-{product}' for product in products]
    dataset_names.extend(['dbgo', 'music',])
    loo_dataset_names = [dn for dn in dataset_names if dn != dataset_name]
    loo_dataset_dirs = [f'data/prepared/{dn}' for dn in loo_dataset_names]
    return loo_dataset_dirs


parser = argparse.ArgumentParser(description='The fast leave one out experiment.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--tbs', type=int, default=1)
parser.add_argument('--vbs', type=int, default=1)
parser.add_argument('--base_model', type=str, 
                    default='/scratch/mhussein/.cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659')
parser.add_argument('--leaved_dataset_name', type=str, default='abt')
parser.add_argument('--serialization_mode', type=str, default='mode1')
parser.add_argument('--output', type=str, default='output',help="output folder")
parser.add_argument('--exp_method', type=str, default='method',help="experiment used in outputing folder")
parser.add_argument('--row_sample_func', type=str, default='automl_filter')
parser.add_argument('--train_data', type=str, default='row', choices=['row', 'attr+row', 'attr-row'])
parser.add_argument('--patience_start', type=int, default=20)
args = parser.parse_args()

seed = args.seed
base_model = args.base_model
leaved_dataset_name = args.leaved_dataset_name
serialization_mode = args.serialization_mode
row_sample_func = args.row_sample_func
train_data = args.train_data
patience_start = args.patience_start

model, tokenizer = load_model(base_model)
dataset_dirs = get_loo_dirs(leaved_dataset_name)

if base_model == 't5-base':
    lr = 1e-4
    DatasetClass = T5Dataset
elif base_model == 'gpt2':
    lr = 2e-5
    DatasetClass = GPTDataset
elif base_model == 'bert-base':
    lr = 2e-5
    DatasetClass = BertDataset
elif 'Llama' in base_model :
    lr = 2e-5
    DatasetClass = LlamaDataset
else:
    raise ValueError('Model not found.')

tbs = args.tbs
vbs = args.vbs
print('-----' * 10)
print(f'Experiment to leave the {leaved_dataset_name} dataset out with {train_data} as training data.', flush=True)
if train_data == 'attr-row':
    print('The model firstly be pre-trained on the attribute pairs to get familiar with the EM task.', flush=True)
    train_attr_df, valid_attr_df, _ = read_multi_attr_data(dataset_dirs, serialization_mode)
    train_attr_d = DatasetClass(tokenizer, train_attr_df, max_len=350)
    valid_attr_d = DatasetClass(tokenizer, valid_attr_df, max_len=350)
    best_model = train(tokenizer, model, train_attr_d, valid_attr_d, epochs=50, lr=lr, seed=seed, patient=True,
                       save_model=False, save_freq=50, train_batch_size=tbs, valid_batch_size=vbs, save_model_path='',
                       save_result_prefix='', patience=6, patience_start=10, base_model=base_model)
    model = copy.deepcopy(best_model)
    print('The pre-training phase is finished.', flush=True)
    dataset_names = ['abt', 'amgo', 'beer', 'dbac', 'dbgo', 'foza', 'itam', 'waam', 'wdc']
    dataset_names = [f'wdc-{product}' for product in products]
    dataset_names.extend(['dbgo', 'music',])
    for dn in dataset_names:
        _, _, test_df = read_single_row_data(f'data/prepared/{dn}', serialization_mode, print_info=False)
        test_d = DatasetClass(tokenizer, test_df, max_len=10000)
        test_f1, test_acc = inference(tokenizer, model, test_d, batch_size=vbs, base_model=base_model)
        print(f'Test acc and f1 after pretraining for {dn} are {test_acc*100:.2f} and {test_f1 * 100:.2f}', flush=True)

    print('Then the model will be fine-tuned on the row level data.', flush=True)
    train_df, valid_df, _ = read_multi_row_data(dataset_dirs, serialization_mode, row_sample_func)

elif train_data == 'row':
    print('The model will be trained on the row level data.', flush=True)
    train_df, valid_df, _ = read_multi_row_data(dataset_dirs, serialization_mode, row_sample_func)

elif train_data == 'attr+row':
    print('The model will be trained on the mixture of attribute and row level data.', flush=True)
    train_attr_df, _, _ = read_multi_attr_data(dataset_dirs, serialization_mode)
    train_row_df, valid_row_df, _ = read_multi_row_data(dataset_dirs, serialization_mode, row_sample_func)
    train_df = pd.concat([train_attr_df, train_row_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    valid_df = valid_row_df

train_d = DatasetClass(tokenizer, train_df, max_len=350)
valid_d = DatasetClass(tokenizer, valid_df, max_len=350)
print('The training phase starts from here.', flush=True)
print(f'The size of the training and validation datasets are: {len(train_d)}, {len(valid_d)}', flush=True)
print(f'Here is the configuration for the experiment:\n'
      f'\tseed: {seed}\tbase_model: {base_model}\tdataset_name: {leaved_dataset_name}\tmode: {serialization_mode} '
      f'\tmax_len: {350}\tlr: {lr}\tbatch_size: {tbs}\tpatience: {6}\tp_start: {patience_start}', flush=True)
best_model = train(tokenizer, model, train_d, valid_d, epochs=50, lr=lr, seed=seed, patient=True, save_model=False,
                   save_freq=50, train_batch_size=tbs, valid_batch_size=vbs, save_model_path='',
                   save_result_prefix='', patience=6, patience_start=patience_start, base_model=base_model)
print('The training phase is finished.', flush=True)

print('Start the evaluation phase.')
dataset_names = ['abt', 'amgo', 'beer', 'dbac', 'dbgo', 'foza', 'itam', 'waam', 'wdc']
dataset_names = [f'wdc-{product}' for product in products]
dataset_names.extend(['dbgo', 'music',])
# Open the file in write mode ('w')
f1_acc_map={}
acc_map={}
for dn in dataset_names:
    _, _, test_df = read_single_row_data(f'data/prepared/{dn}', serialization_mode, print_info=False)
    test_d = DatasetClass(tokenizer, test_df, max_len=10000)
    test_f1, test_acc = inference(tokenizer, best_model, test_d, batch_size=vbs, base_model=base_model)
    f1_acc_map[dn]=test_f1*100
    acc_map[dn]=test_acc*100
    print(f'Test acc and f1 for {dn} are {test_acc*100:.2f} and {test_f1*100:.2f}', flush=True)
output_path =os.path.join(os.getcwd(),os.path.join(args.output,args.exp_method))
if not(os.path.exists(output_path) and os.path.isdir(output_path)):
    os.makedirs(output_path)
with open(os.path.join(output_path,f'{args.base_model}_{args.leaved_dataset_name}_{args.train_data}.txt'), 'w') as file:
    file.write(f"All arguments:: {vars(args)}\n")
    file.write(f"Evaluation Results\n")
    for dn in f1_acc_map.keys():
        file.write(f"{dn}:: F1:: {f1_acc_map[dn]}  , Test ACC:: {acc_map[dn]} \n")

print('Evaluation finished.', flush=True)

print('-----' * 10)

