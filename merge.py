#this script merges the data from list of different files into one file
#6c2974c9-59f8-4d07-914a-cc3d3d9c4c2f.jpg,['4', '4', '4', '4', '4']
import os
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--output-dir', type=str, default='/home/hvtham/LAVA/son/output')

def merge_parts(part='Public', output_dir='/home/hvtham/LAVA/son/output'):

    list = [out_vote for out_vote in os.listdir(output_dir) if 'vote' in out_vote and part in out_vote]

    weights = [1 if '7B' in out_vote else 2 for out_vote in list]

    output_file = os.path.join(output_dir, f'{part}_assemble.csv')
    list_files = []
    for file in list:
        with open(os.path.join(output_dir, file), 'r') as f:
            lines = f.readlines()
            list_files.append(lines)
    votes = {}
    for i,file in enumerate(list_files):
        for line in file:
            line = line.split(',')
            if line[0] not in votes:
                votes[line[0]] = []
            list_vote = ''.join(line[1:]).replace('[', '').replace(']', '').replace('\n', '').replace('\'', '')
            list_vote = [int(i) for i in list_vote.split(' ')]
            list_vote = weights[i] * list_vote
            votes[line[0]].extend(list_vote)

    with open(output_file, 'w') as f:
        f.write("file_name,answer\n")
        for key, value in votes.items():
            votes[key] = np.argmax(np.bincount(value))
            f.write(f"{key},{votes[key]}\n")

if __name__ == '__main__':
    args = parser.parse_args()
    merge_parts(part='Public', output_dir=args.output_dir)
    merge_parts(part='Private', output_dir=args.output_dir)