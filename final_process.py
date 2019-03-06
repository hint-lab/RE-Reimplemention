import os
from helper import *
import json
from collections import defaultdict as ddict
import pdb
rel2id=json.loads(open('./data/rel2id.json').read())
id2rel=dict([(v,k) for k,v in rel2id.items()])
data={"train":[],"test":[]}

def read_file(file_path,split):
    with open(file_path) as f:
        for k,line in enumerate(f):
            bag=json.loads(line.strip())
        
            pos1_list=[]
            pos2_list=[]
            head_pos_list=[]
            tail_pos_list=[]
            
            for sent in bag['sentence']:
                tid_map,tid2wrd=ddict(dict),ddict(list)
                tok_idx=1
                head_pos,tail_pos=None,None
                i=0
                tokens=sent['sent'].split(' ')
                while i < len(tokens):
                    if tokens[i]==bag['head']:
                        head_pos=tok_idx-1
                    if tokens[i]==bag['tail']:
                        tail_pos=tok_idx-1
                    tok_idx+=1
                    i+=1
                if head_pos==None or tail_pos==None:
                    print('entry skipped!!!')
                    print("{}|{}|{}".format(bag['head'],bag['tail'],sent['sent']))
                    continue
                pos1=[i-head_pos for i in range(tok_idx-1)]
                pos2=[i-tail_pos for i in range(tok_idx-1)]
                pos1_list.append(pos1)
                pos2_list.append(pos2)
                head_pos_list.append(head_pos)
                tail_pos_list.append(tail_pos)
            data[split].append({
                'head':bag['head'],
                'tail':bag['tail'],
                'head_pos_list':head_pos_list,
                'tail_pos_list':tail_pos_list,
                'pos1_list':pos1_list,
                'pos2_list':pos2_list,
            })
            if k%1000 ==0:
                print('completed{}'.format(k))

read_file('./data/train_bags.json','train')
