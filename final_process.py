import os
from helper import *
import json
from collections import defaultdict as ddict
import pdb
import re
import unicodedata
import nltk
from tqdm import tqdm
rel2id = json.loads(open("./data/rel2id.json",encoding='utf-8').read())
id2rel = dict([(v, k) for k, v in rel2id.items()])
data = {"train": [], "test": []}



def read_file(file_path):
    temp = []
    with open(file_path,encoding='utf-8') as f:
        for k, line in enumerate(f):
            bag = json.loads(line.strip())

            pos1_list = []
            pos2_list = []
            head_pos_list = []
            tail_pos_list = []
            wrds_list = []


            # print('complete substitution')

            for sent in bag["sentence"]:#分词
                # sent["nlp"]=ddict({"tokens":list})
                sent["nlp"] = {}
                sent["nlp"]["sentences"] = []
                tokenlist = []
                tokens = sent["sent"].split()
                for index, word in enumerate(tokens):
                    token = {}
                    token["index"] = index
                    token["originalText"] = word
                    token["characterOffsetBegin"] = len(" ".join(sent["sent"].split()[0:index])) + (
                        1 if index != 0 else 0)
                    token["characterOffsetEnd"] = len(" ".join(sent["sent"].split()[0:index])) + len(word) + (
                        1 if index != 0 else 0)
                    tokenlist.append(token)
                sent["nlp"]["sentences"].append({"tokens": tokenlist})

            # print('complete adding nlp')
            #print(bag)
            count=0

            #debug
            # if k>=90000:
            #     print(len(bag['sentence']))
            #     print(bag)

            for sent in bag["sentence"]:
                #debug
                # if k>=90000:
                #     print(sent)

                # 输出head词和tail词在句子中的索引位置到(list)head_start_off和(list)tail_start_off，先找一个词的索引位置，再找另一个词，且另一个词的索引位置必须在第一个词
                # 的第一个字之前或最后一个字之后
                # 实体词由一个字或两个字组成，先分为len(head)>len(tail)和len(head)<=len(tail)
                if len(bag["head"]) > len(bag["tail"]):  # head词比tail词长的时候
                    head_idx = [i for i, e in enumerate(sent["sent"].split()) if
                                e == bag["head"]]  # 不计词间空格时的head词的词列表索引位置（考虑多个head词）
                    head_start_off = [len(" ".join(sent["sent"].split()[0:idx])) + (1 if idx != 0 else 0) for idx in
                                      head_idx]  # 计入词间空格时head词的句子列表索引位置（假设idx=0时，head_start_off=0;idx=1时，head_start_off=2)(对中文分词有利）
                    if head_start_off == []:  # 如果head是两个字的词时，用下划线取代空格后，利用正则表达式匹配
                        head_start_off = [
                            m.start() for m in re.finditer(
                                bag["head"].replace("_", " "),
                                sent["sent"].replace("_", " ")
                            )
                        ]
                    reserve_span = [(start_off, start_off + len(bag["head"]))
                                    for start_off in head_start_off]  # head词的span，(第一个字的索引位置，最后一个字的索引位置）

                    tail_idx = [i for i, e in enumerate(sent["sent"].split()) if e == bag["tail"]]
                    tail_start_off = [len(" ".join(sent["sent"].split()[0:idx])) + (1 if idx != 0 else 0) for idx in
                                      tail_idx]
                    if tail_start_off == []:
                        tail_start_off = [
                            m.start() for m in re.finditer(
                                bag["tail"].replace("_", " "),
                                sent["sent"].replace("_", " ")
                            )
                        ]
                    tail_start_off = [
                        off for off in tail_start_off if all([
                            off < span[0] or off > span[1]
                            for span in reserve_span
                        ])
                    ]  # 筛选tail_start_off，tail词的句子列表索引位置,必须满足在head词的第一个字之前，或在最后一个字之后
                else:  # head词和tail词一样长，或head词短于tail词
                    tail_idx = [
                        i for i, e in enumerate(sent["sent"].split()) if e == bag["tail"]
                    ]
                    tail_start_off = [
                        len(" ".join(sent["sent"].split()[0:idx])) + (1 if idx != 0 else 0) for idx in tail_idx
                    ]
                    if tail_start_off == []:  # 把句子中的空格替换成下划线后再查找实体位置，start()返回的是pattern开始的位置
                        tail_start_off = [
                            m.start() for m in re.finditer(
                                bag["tail"].replace("_", " "),
                                sent["sent"].replace("_", " ")
                            )
                        ]
                    reserve_span = [(start_off, start_off + len(bag["tail"]))
                                    for start_off in tail_start_off]  # tail词的span
                    head_idx = [
                        i for i, e in enumerate(sent["sent"].split()) if e == bag["head"]
                    ]
                    head_start_off = [
                        len(" ".join(sent["sent"].split()[0:idx])) + (1 if idx != 0 else 0) for idx in head_idx
                    ]
                    if head_start_off == []:
                        head_start_off = [
                            m.start() for m in re.finditer(
                                bag["head"].replace("_", " "),
                                sent["sent"].replace("_", " ")
                            )
                        ]
                    head_start_off = [
                        off for off in head_start_off if all([
                            off < span[0] or off > span[1]
                            for span in reserve_span
                        ])
                    ]
                #'词span元组[(开始位置,结束位置,"词名"),...]')
                head_off = [(head_off, head_off + len(bag["head"]), "head")
                            for head_off in head_start_off]
                tail_off = [(tail_off, tail_off + len(bag["tail"]), "tail")
                            for tail_off in tail_start_off]
                if head_off == [] or tail_off == []:
                    continue
                spans = [head_off[0]] + [tail_off[0]]
                off_begin, off_end, _ = zip(*spans)

                tid_map, tid2wrd = ddict(dict), ddict(list)

                tok_idx = 1
                head_pos, tail_pos = None, None




                for s_n, sentence in enumerate(sent["nlp"]["sentences"]):
                    i, tokens = 0, sentence["tokens"]
                    while i < len(tokens):
                        #print('sent order {}'.format(i))
                        if tokens[i]['characterOffsetBegin'] in off_begin:
                            _, end_offset, identity = spans[off_begin.index(tokens[i]['characterOffsetBegin'])]

                            if identity == 'head':
                                head_pos = tok_idx - 1  # Indexing starts from 0
                                tok_list = [tok['originalText'] for tok in tokens]
                            else:
                                tail_pos = tok_idx - 1
                                tok_list = [tok['originalText'] for tok in tokens]

                            while i < len(tokens) and tokens[i]['characterOffsetEnd'] <= end_offset:
                                tid_map[s_n][tokens[i]['index']] = tok_idx
                                tid2wrd[tok_idx].append(tokens[i]['originalText'])
                                i += 1

                            tok_idx += 1
                        else:
                            tid_map[s_n][tokens[i]['index']] = tok_idx
                            tid2wrd[tok_idx].append(tokens[i]['originalText'])

                            i += 1
                            tok_idx += 1


                if head_pos == None or tail_pos == None:
                    print('Skipped entry!!')
                    print('{} | {} | {}'.format(bag['head'], bag['tail'], sent['sent']))
                    continue

                wrds = ['_'.join(e).lower() for e in tid2wrd.values()]
                pos1 = [i - head_pos for i in range(tok_idx - 1)]  # tok_id = (number of tokens + 1)
                pos2 = [i - tail_pos for i in range(tok_idx - 1)]

                wrds_list.append(wrds)
                pos1_list.append(pos1)
                pos2_list.append(pos2)
                head_pos_list.append(head_pos)
                tail_pos_list.append(tail_pos)
                count+=1


            temp.append({
                'head': bag['head'],
                'tail': bag['tail'],
                'rels': bag['relation'],
                # 'phrase_list':		phrase_list,
                'head_pos_list': head_pos_list,
                'tail_pos_list': tail_pos_list,
                'wrds_list': wrds_list,
                'pos1_list': pos1_list,
                'pos2_list': pos2_list,
                # 'sub_type': ent2type[bag['sub_id']],
                # 'obj_type': ent2type[bag['obj_id']],
                # 'dep_links_list':	dep_links_list,
            })

            if k%10000==0:print('Completed {}'.format(k))
            # if not args.FULL and k > args.sample_size: break
    return temp


data['train'] = read_file("./data/train_bags.json")
data['test'] = read_file("./data/test_bags.json")
print('Bags processed:Train:{},Test:{}'.format(len(data['train']),len(data['test'])))

"""*************************************删除离群数据****************************************"""

del_cnt = 0
MAX_WORDS = 100
for dtype in ['train', 'test']:
    for i in range(len(data[dtype]) - 1, -1, -1):
        bag = data[dtype][i]

        for j in range(len(bag['wrds_list']) - 1, -1, -1):
            data[dtype][i]['wrds_list'][j] = data[dtype][i]['wrds_list'][j][:MAX_WORDS]
            data[dtype][i]['pos1_list'][j] = data[dtype][i]['pos1_list'][j][:MAX_WORDS]
            data[dtype][i]['pos2_list'][j] = data[dtype][i]['pos2_list'][j][:MAX_WORDS]

        if len(data[dtype][i]['wrds_list']) == 0:
            del data[dtype][i]
            del_cnt += 1
            continue

print('Bags deleted {}'.format(del_cnt))

"""***********************************建立词库**********************************************"""
MAX_VOCAB=150000
#词频字典
voc_freq=ddict(int)

for bag in data['train']:
    for sentence in bag['wrds_list']:
        for wrd in sentence:
            voc_freq[wrd]+=1

freq=list(voc_freq.items())
freq.sort(key=lambda  x:x[1],reverse=True)
freq=freq[:MAX_VOCAB]
vocab,_=map(list,zip(*freq))

vocab.append('UNK')

"""*******************************建立word 和 id之间的映射表*********************************"""

#词到id的字典
def getIdMap(vals,begin_idx=0):
    ele2id={}
    for id,ele in enumerate(vals):
        ele2id[ele]=id+begin_idx
    return ele2id

voc2id=getIdMap(vocab,1)
id2voc=dict([(v,k) for k,v in voc2id.items()])

print('Chosen Vocabulary:\t{}'.format(len(vocab)))

"""******************************将数据转化为张量形式************************************"""

MAX_POS=60#并不是最终的max_pos,而是计算max_pos的margin
#词转id
def getId(wrd,wrd2id,def_val='NONE'):
    if wrd in wrd2id:
        return wrd2id[wrd]
    else:
        return wrd2id[def_val]

def posMap(pos):
    if pos<-MAX_POS:
        return 0
    elif pos > MAX_POS:
        return (MAX_POS+1)*2
    else:
        return pos+(MAX_POS+1)

def procData(data,split='train'):
    result=[]

    for bag in data:
        res={}#k-hot label
        res['X']=[[getId(wrd,voc2id,'UNK') for wrd in wrds] for wrds in bag['wrds_list']]
        res['Pos1']=[[posMap(pos) for pos in pos1] for pos1 in bag['pos1_list']]
        res['Pos2']=[[posMap(pos) for pos in pos2] for pos2 in bag['pos2_list']]
        res['Y']=bag['rels']
        res['HeadPos']=bag['head_pos_list']
        res['TailPos']=bag['tail_pos_list']
        result.append(res)
    return result

final_data={
    "train":procData(data['train'],'train'),
    "test":procData(data['test'],'test'),
    "voc2id":voc2id,
    "id2voc":id2voc,
    "max_pos":(MAX_POS+1)*2+1,
    "rel2id":rel2id
}
print('writing final_data')
pickle.dump(final_data,open("{}_processed.pkl".format("riedel"),'wb'))