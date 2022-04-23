import argparse
import os
import sys

from BiLSTM.Main import main as main_bilstm
# from DAN.Main import main as main_dan
from Transformer.Main import main as main_transformer


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--tag_feature', type=str, required=True)
parser.add_argument('--dataset_size', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

# if len(sys.argv) != 5:
#     assert False, "Error - Didn't pass required arguments"
# dataset_name = sys.argv[1]
# tag_feature = sys.argv[2]  # tags = 'original' / 'upos' / ...
# limit = int(sys.argv[3])
# model = sys.argv[4]

possible_datasets = [
    'IMDB_Dataset_5000.json',
    'IMDB_Dataset_15000.json',
    'News_Dataset_5000.json',
    'News_Dataset_15000.json'
]
limit = args.dataset_size

list_datasets = os.listdir('./Datasets')
exist_datasets = [dataset for dataset in possible_datasets if dataset in list_datasets and args.dataset_name in dataset]
if len(exist_datasets) == 0:
    assert False, "There's no relevant prepared dataset in Datasets folder."

if args.tag_feature not in ['original', 'upos', 'upos_filter', 'upos_filter_extend',
                       'bigram', 'bigram', 'bigram_filter', 'bigram_filter_extend']:
    assert False, "Error - Not supported preprocessing method"

if args.model not in ['BiLSTM', 'DAN', 'Transformer']:
    assert False, "Error - Not supported model"

if args.dataset_size == 15000 and [dataset for dataset in exist_datasets if str(args.dataset_size) in dataset] == 0:
    print("There's no relevant prepared dataset with size 15000 -> limit was set to 5000")
    limit = 5000

if args.dataset_size not in [15000, 5000, 1000]:
    print("Not supported limit size -> limit was set to 5000")
    limit = 5000

dataset_size = limit if limit != 1000 else 5000

if args.model == 'BiLSTM':
    main_bilstm(args.dataset_name, args.tag_feature, limit, dataset_size)

# if args.model == 'DAN':
#     main_dan(args.dataset_name, args.tag_feature, limit, dataset_size)

if args.model == 'Transformer':
    main_transformer(args.dataset_name, args.tag_feature, limit, dataset_size)

print('~~~~~~    Finish    ~~~~~~')
