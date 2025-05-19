import wandb
import argparse
import json 
import os 
import itertools 

# Parse the arguments
def get_parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CoopSheaf')#, required=True)
    parser.add_argument('--project', type=str, default='coopshv_sweeps')
    parser.add_argument('--datasets', type=str, default='multiclass')
    return parser 

MULTICLASS_DATA = ['amazon_ratings', 'roman_empire'] #multiclass
BINARY_DATA = ['minesweeper', 'tolokers', 'questions'] #binary classification (maybe lower lr on these)
# we have some memory allocation issues on tolokers and questions depending on the combination of
# parameters

def get_sweep_filename(dataset): 
    return f'sweeps_config/model_sweep_id_{dataset}.json'

def get_model_list(args): 
    # Check if multiple models were passed as arguments 
    if ',' not in args.model: 
        return [args.model] 
    else: 
        return [model for model in args.model.split(',')]   
    
def create_sweep_config(args, dataset): 
    # Define the sweep configuration
    sweep_config = {
        'program': 'exp/run.py',
        'name': f'{args.model}_{dataset}',
        'method': 'grid', # we could also use random
        'metric': {
            'goal': 'maximize',
            'name': 'val_acc'
        },
        'parameters': {
            'dataset': {
                'values': [dataset] 
            },
            'd': {
                'values': [4,5]
            },
            'hidden_channels': {
                'values': [32,64]
            },
            'epochs': {
                'value': 2000 #I found the best accuracies using 3k epochs with high/no early stopping (only for amazon, roman and minesweeper, manual tunning)
            },
            'early_stopping': {
                'value': 200
            },
            'layers': {
                'values': [2,3,4,5]
            },
            'gnn_layers': {
                'values': [0,1,2,3,4,5]
            },
            'gnn_hidden': {
                'values': [32,64] 
            },
            'pe_size': {
                'value': 0
            },
            'left_weights': {
                'values': [True, False]
            },
            'right_weights': {
                'values': [True, False]
            },
            'lr': {
                'value': 0.02 #maybe fixed 0.02 for amazon and roman, 2e-3, 2e-4 or 2e-5 for the rest
            },
            'lr_decay_patience': {
                'value': 20 #we're not using scheduler rn
            },
            'weight_decay': {
                'values': [1e-7, 1e-8] #found 1e-7 and 1e-8 to be good in some datasets
            },
            'input_dropout': {
                'value': 0.2
            },
            'dropout': {
                'value': 0.2
            },
            'orth': {
                'value': 'householder'
            },
            'use_act': {
                'value': True
            },
            'model': {
                'value': args.model 
            },
            'edge_weights': {
                'value': False
            },
            'normalised': {
                'value': True
            },
            'sparse_learner': {
                'value': True
            },
            'entity': {
                'value': 'andrerg00'
            }
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity='andrerg00')
    print(f'Sweep ID for {args.model}: {sweep_id}')
    return sweep_id 

if __name__ == '__main__': 
    parser = get_parser() 
    args = parser.parse_args()

    model_lst = get_model_list(args)
    datasets = MULTICLASS_DATA if args.datasets == 'multiclass' else BINARY_DATA
    for dataset in datasets: 
        filename = get_sweep_filename(dataset) 
        sweep_dct = dict() 
        if not os.path.exists('sweeps_config'): 
            os.mkdir('sweeps_config') 
        if os.path.exists(filename): 
            sweep_dct = json.load(open(filename)) 
        for model in model_lst: 
            args.model = model 
            sweep_id = create_sweep_config(args, dataset) 
            sweep_dct[model] = sweep_id    
        json.dump(
            sweep_dct, open(filename, 'w')
        )