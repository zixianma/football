import pandas as pd
import os 
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('--input-dir', type=str, default='/vision2/u/zixianma/football_results', required=True)
parser.add_argument('--output-dir', type=str, default='/vision2/u/zixianma/football_results/data')
parser.add_argument('--seeds', type=int, nargs='+', help='a list of random seeds', required=True)
parser.add_argument('--eval-start', type=int, default=0)
parser.add_argument('--eval-end', type=int, default=50000)
parser.add_argument('--metric', type=str, default='avg')
args = parser.parse_args()

dirpath = args.input_dir
metric_vals = []
start, end = args.eval_start, args.eval_end
for i in args.seeds:
    filename = f'seed{i}.json'
    json_path = os.path.join(os.path.join(dirpath, filename))
    print(json_path)
    df = pd.read_json(json_path, lines=True)
    if args.metric == 'avg':
        metric_vals.append(np.mean(df['episode_reward_mean'][start:end]))
    elif args.metric == 'max':
        metric_vals.append(np.max(df['episode_reward_mean'][start:end]))
    else:
        raise NotImplementedError

divide_pos = args.input_dir.rfind('/')
exp_name = args.input_dir[:divide_pos]
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
output_path = os.path.join(args.output_dir, exp_name)

data = {}
mu = np.mean(metric_vals)
sigma = np.std(metric_vals) / np.sqrt(len(metric_vals))
data[args.metric + '_mean'] = mu
data[args.metric + '_stderr'] = sigma
output_df = pd.DataFrame(data)
output_df.to_csv(output_path, index=False)