import pandas as pd
import os 
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('--input-dir', type=str, default='/vision2/u/zixianma/football_results', required=True)
parser.add_argument('--output-dir', type=str, default='/vision2/u/zixianma/football_results/data')
parser.add_argument('--seeds', type=int, nargs='+', help='a list of random seeds', required=True)
parser.add_argument('--eval-last', type=int, default=50000)
# parser.add_argument('--eval-end', type=int, default=50000)
parser.add_argument('--metric', type=str, default='avg')
args = parser.parse_args()

dirpath = args.input_dir
metric_vals = []

for i in args.seeds:
    filename = f'seed{i}.json'
    json_path = os.path.join(os.path.join(dirpath, filename))
    print(json_path)
    df = pd.read_json(json_path, lines=True)
    end = len(df)
    assert end >= args.eval_last, "There aren't enough entries in the input file. \
    Decrease the eval-last number or double check your input file."
    start = end - args.eval_last
    print(f'The are {len(df)} entries in the input file.')
    if args.metric == 'avg':
        avg = np.mean(df['episode_reward_mean'][start:end])
        print(f'The avg episode reward for seed {i} is {avg}.')
        metric_vals.append(avg)
    elif args.metric == 'max':
        max = np.max(df['episode_reward_mean'][start:end])
        print(f'The max episode reward for seed {i} is {max}.')
        metric_vals.append(max)
    else:
        raise NotImplementedError

divide_pos = args.input_dir.rfind('/')
exp_name = args.input_dir[divide_pos+1:] + '.csv'
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
output_path = os.path.join(args.output_dir, exp_name)
print(f"Outputing to {output_path}...")
data = {}
mu = np.mean(metric_vals)
sigma = np.std(metric_vals) / np.sqrt(len(metric_vals))
print(f"The mean and stderr are {mu} and {sigma}.")
data['exp_name'] = [exp_name + f'_{args.metric}_last{args.eval_last}']
data[args.metric + '_mean'] = [mu]
data[args.metric + '_stderr'] = [sigma]
output_df = pd.DataFrame(data)
output_df.to_csv(output_path, index=False)
