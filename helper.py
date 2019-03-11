import tensorflow as tf
import numpy as np
import os,sys,json,random,argparse
import logging,logging.config,pathlib
import pickle,uuid,time,pdb,gensim,itertools
from collections import defaultdict as ddict
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score
from pprint import pprint
#所有文件的库都在这里导入

#设置numpy的精度
np.set_printoptions(precision=4)

#从w2v文件中读出embeddings
def getEmbeddings(model,wrd_list,embed_dims):
    embed_list=[]
    for word in wrd_list:
        if word in model.vocab:
            embed_list.append(model.word_vec(word))
        else:
            embed_list.append(np.random.randn(embed_dims))
    return np.array(embed_list,dtype=np.float32)

#gpu设置
def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus

#检查路径下文件是否存在
def checkFile(filename):
    return pathlib.Path(filename).is_file()

#创建路径
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

#tensorflow模型debug
def debug_nn(res_list,feed_dict):
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess=tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    summary_writer=tf.summary.FileWriter("tf_board/debug_nn",sess.graph)
    res=sess.run(res_list,feed_dict=feed_dict)
    return res

#创建日志器
def get_logger(name,log_dir,config_dir):
    make_dir(log_dir)
    config_dict=json.load(open(config_dir+'log_config.json'))
    config_dict['handlers']['file_handler']['filename']=log_dir+name.replace('/','-')
    logging.config.dictConfig(config_dict)
    logger=logging.getLogger(name)

    std_out_format='%(asctime)s-[%(levelname)s]-%(message)s'
    consoleHandler=logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

#将输入列表分段为相同长度的多个列表
def getChunks(inp_list,chunk_size):
    return [inp_list[x:x+chunk_size] for x in range(0,len(inp_list),chunk_size)]