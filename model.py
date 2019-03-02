import tensorflow as tf
import numpy as np
import json,time
from collections import defaultdict
#读取原始数据
relation2id = json.loads(open('./data/rel2id.json').read())
train_data = defaultdict(lambda: {'rels': defaultdict(list)})
miss_cnt=0
count=0
with open('./data/train.json', 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line.strip())
        _id = '{}_{}'.format(data['head']['word'], data['tail']['word'])
        train_data[_id]['head_id'] = data['head']['id']
        train_data[_id]['tail_id'] = data['tail']['id']
        train_data[_id]['head'] = data['head']['word']
        train_data[_id]['tail'] = data['tail']['word']

        train_data[_id]['rels'][relation2id.get(data['relation'],
                                                relation2id['NA'])].append(
                                                    {'sent': data['sentence']})
        if i%10000==0:
            print('reading raw train data completed {}/{},{}'.format(i,miss_cnt,time.strftime("%d_%m_%Y")+'_'+time.strftime("%H:%M:%S")))
# print(train_data)
test_data=defaultdict(lambda:{'sents':[],'rels':set()})
with open('./data/test.json','r') as f:
    for i,line in enumerate(f):
        data=json.loads(line.strip())
        _id='{}_{}'.format(data['head']['word'],data['tail']['word'])
        test_data[_id]['head_id']=data['head']['id']
        test_data[_id]['tail_id']=data['tail']['id']
        test_data[_id]['head']=data['head']['word']
        test_data[_id]['tail']=data['tail']['word']
        test_data[_id]['rels'].add(relation2id.get(data['relation'],relation2id['NA']))
        test_data[_id]['sents'].append({'sent':data['sentence']})
        if i%10000==0:
            print('reading raw test data completed {}/{},{}'.format(i,miss_cnt,time.strftime("%d_%m_%Y")+'_'+time.strftime("%H:%M:%S")))
#将原始数据按包分组
# print(test_data)
count=0
with open('./data/train_bags.json','w') as f:
    for _id,data in train_data.items():
        for rel,sents in data['rels'].items():

            entry={}
            entry['head']=data['head']
            entry['tail']=data['tail']
            entry['head_id']=data['head_id']
            entry['tail_id']=data['tail_id']
            entry['sentence']=sents
            entry['relation']=[rel]

            f.write(json.dumps(entry,ensure_ascii=False)+'\n')
            count+=1
            if count%10000==0:
                print("writing train bags completed {},{}".format(count,time.strftime("%d_%m_%Y")+'_'+time.strftime("%H:%M:%S")))

count=0
with open('./data/test_bags.json','w') as f:
    for _id,data in test_data.items():

        entry={}
        entry['head']=data['head']
        entry['tail']=data['tail']
        entry['head_id']=data['head_id']
        entry['tail_id']=data['tail_id']
        entry['sentence']=data['sents']
        entry['rel']=list(data['rels'])

        f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        count+=1
        if count%10000==0:
            print('Writing test bags completed {},{}'.format(count,time.strftime("%d_%m_%Y")+'_'+time.strftime("%H:%M:%S")))