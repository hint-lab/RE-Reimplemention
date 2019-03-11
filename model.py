import tensorflow as tf
from helper import *


class Model(object):

    def __init__(self, params):
        self.params = params
        self.logger = get_logger(self.params.name, self.params.log_dir, self.params.config_dir)
        self.logger.info(vars(self.params))
        pprint(vars(self.params))
        self.params.batch_size = self.params.batch_size

        if self.params.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.params.l2)

        self.load_data()
        self.add_placeholder()

        nn_out, self.accuracy = self.add_model()

        self.loss = self.add_loss(nn_out)
        self.logits = tf.nn.softmax(nn_out)
        self.train_op = self.add_optimizer(self.loss)

        tf.summary.scalar('accuracy', self.accuracy)
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = None

    def load_data(self):
        data = pickle.load(open(self.params.dataset, 'rb'))

        self.voc2id = data['voc2id']
        self.id2voc = data['id2voc']
        self.max_pos = data['max_pos']
        self.num_class = len(data['rel2id'])
        self.num_deLabel = 1

        # get word list
        self.word_list = list(self.voc2id.items())
        self.word_list.sort(key=lambda x: x[1])
        self.word_list, _ = zip(*self.word_list)

        self.test_one, \
        self.test_two = self.getPNdata(data)

        self.data = data
        # self.data=self.splitBags(data,self.params.chunk_size) #acitivate when bags are too big

        self.logger.info(
            'Document count [{}]:{}, [{}]:{}'.format('trian', len(self.data['train']), 'test', len(self.data['test'])))

    # 为计算图添加占位符变量
    def add_placeholder(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_data')
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name='input_label')
        self.input_pos1 = tf.placeholder(tf.int32, shape=[None, None], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, shape=[None, None], name='input_pos2')
        self.mask = tf.placeholder(tf.int32, shape=[None, None], name='mask')
        self.x_len = tf.placeholder(tf.int32, shape=[None], name='input_len')  # 输入句子的词数
        self.sent_num = tf.placeholder(tf.int32, shape=[None, 3],
                                       name='sent_num')  # stores which sentences belong to which bags
        self.seq_len = tf.placeholder(tf.int32, shape=(), name='seq_len')  # 记录batch里句子的最大长度
        self.total_bags = tf.placeholder(tf.int32, shape=(), name='total_bags')  # 一个batch里包的数量
        self.total_sents = tf.placeholder(tf.int32, shape=(), name='total_sents')  # 一个batch里句子的数量
        self.dropout = tf.placeholder_with_default(self.params.dropout, shape=(), name='dropout')  # 用于全连接层的dropout

    def getBatches(self, data, shuffle=True):
        # 创建以包为单位的batch
        if shuffle: random.shuffle(data)
        for chunk in getChunks(data, self.params.batch_size):  # chunk=batch
            batch = ddict(list)
            num = 0
            for i, bag in enumerate(chunk):
                batch['X'] += bag['X']
                batch['Pos1'] += bag['Pos1']
                batch['Pos2'] += bag["Pos2"]
                batch['Mask'] += bag['Mask']
                batch['Y'].append(bag['Y'])
                old_num = num
                num += len(bag['X'])
                batch['sent_num'].append([old_num, num, i])

            yield batch

    def splitBags(self, data, chunk_size):
        # 将大于batch_size的包分割
        for dtype in ['train']:
            for i in range(len(data[dtype]) - 1, -1, -1):
                bag = data[dtype][i]
                if len(bag['X']) > chunk_size:
                    del data[dtype][i]
                    chunks = getChunks(range(len(bag['X'])), chunk_size)

                    for chunk in chunks:
                        res = {
                            'Y': bag['Y']
                        }
                        res['X'] = [bag['X'][j] for j in chunk]
                        res['Pos1'] = [bag['Pos1'][j] for j in chunk]
                        res['Pos2'] = [bag['Pos2'][j] for j in chunk]
                        res['Mask'] = [bag['Mask'][j] for j in chunk]

                        data[dtype].append(res)

        return data

    def getPNdata(self, data):
        # 为计算p@n准备数据
        p_one = []
        p_two = []

        for bag in data['test']:
            if len(bag['X']) < 2:
                continue
            index = list(range(len(bag['X'])))
            random.shuffle(index)

            p_one.append({
                'X': [bag['X'][index[0]]],
                'Pos1': [bag['Pos1'][index[0]]],
                'Pos2': [bag['Pos2'][index[0]]],
                'Mask': [bag['Mask'][index[0]]],
                'Y': bag['Y']
            })
            p_two.append({
                'X': [bag['X'][index[0]], bag['X'][index[1]]],
                'Pos1': [bag['Pos1'][index[0]], bag['Pos1'][index[1]]],
                'Pos2': [bag['Pos2'][index[0]], bag['Pos2'][index[1]]],
                'Mask': [bag['Mask'][index[0]], bag['Pos2'][index[1]]],
                'Y': bag['Y']
            })
        return p_one, p_two

    def padData(self, data, seq_len):
        # 为句子补位
        temp = np.zeros((len(data), seq_len), np.int32)
        mask = np.zeros((len(data), seq_len), np.float32)

        for i, ele in enumerate(data):
            temp[i, :len(ele)] = ele[:seq_len]
            mask[i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

        return temp, mask

    def pad_dynamic(self, X, pos1, pos2, mask):
        # 为每个batch中的句子补位
        seq_len = 0
        x_len = np.zeros((len(X)), np.int32)

        for i, x in enumerate(X):
            seq_len = max(seq_len, len(x))
            x_len[i] = len(x)

        x_pad, _ = self.padData(X, seq_len)
        pos1_pad, _ = self.padData(pos1, seq_len)
        pos2_pad, _ = self.padData(pos2, seq_len)
        mask_pad, _ = self.padData(mask, seq_len)

        return x_pad, x_len, pos1_pad, pos2_pad, mask_pad, seq_len

    def getOneHot(self, Y, num_class):
        # 将标签转换为one-hot向量形式
        temp = np.zeros((len(Y), num_class), np.int32)
        for i, e in enumerate(Y):
            for rel in e:
                temp[i, rel] = 1

        return temp

    def create_feed_dict(self, batch, wLabels=True, dtype='train'):
        # 创建feed_dict传入计算图
        X, Y, pos1, pos2, mask, sent_num = batch['X'], batch['Y'], batch['Pos1'], batch['Pos2'], batch['Mask'], batch[
            'sent_num']
        total_sents = len(batch['X'])
        total_bags = len(batch['Y'])
        x_pad, x_len, pos1_pad, pos2_pad, mask_pad, seq_len = self.pad_dynamic(X, pos1, pos2, mask)

        y_hot = self.getOneHot(Y, self.num_class)

        feed_dict = {}
        feed_dict[self.input_x] = np.array(x_pad)
        feed_dict[self.input_pos1] = np.array(pos1_pad)
        feed_dict[self.input_pos2] = np.array(pos2_pad)
        feed_dict[self.mask] = np.array(mask_pad)
        feed_dict[self.x_len] = np.array(x_len)
        feed_dict[self.seq_len] = np.array(seq_len)
        feed_dict[self.total_sents] = np.array(total_sents)
        feed_dict[self.total_bags] = np.array(total_bags)
        feed_dict[self.sent_num] = np.array(sent_num)

        if wLabels:
            feed_dict[self.input_y] = y_hot

        if dtype != 'train':
            feed_dict[self.dropout] = 1.0
        else:
            feed_dict[self.dropout] = self.params.dropout

        return feed_dict

    def add_model(self):
        input_words, input_pos1, input_pos2, mask = self.input_x, self.input_pos1, self.input_pos2, self.mask

        with tf.variable_scope('Embeddings') as scope:
            model = gensim.models.KeyedVectors.load_word2vec_format(self.params.embed_loc, binary=False)
            embed_init = getEmbeddings(model, self.word_list, self.params.embed_dim)
            _word_embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True,
                                               regularizer=self.regularizer)
            word_pad = tf.zeros([1, self.params.embed_dim])
            word_embeddings = tf.concat([word_pad, _word_embeddings], axis=0)

            pos1_embeddings = tf.get_variable('pos1_embeddings', [self.max_pos, self.params.pos_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)
            pos2_embeddings = tf.get_variable('pos2_embeddings', [self.max_pos, self.params.pos_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)

        word_embeded = tf.nn.embedding_lookup(word_embeddings, input_words)
        pos1_embeded = tf.nn.embedding_lookup(pos1_embeddings, input_pos1)
        pos2_embeded = tf.nn.embedding_lookup(pos2_embeddings, input_pos2)
        embeds = tf.concat([word_embeded, pos1_embeded, pos2_embeded], axis=2)
        embeds_dim=self.params.embed_dim+2*self.params.pos_dim

        with tf.variable_scope('word_attention') as scope:
            word_query = tf.get_variable('word_query', [embeds_dim, 1],
                                         initializer=tf.contrib.layers.xavier_initializer())
            sent_repre = tf.reshape(
                tf.matmul(
                    tf.reshape(
                        tf.nn.softmax(
                            tf.reshape(
                                tf.matmul(
                                    tf.reshape(tf.tanh(embeds),[self.total_sents*self.seq_len,embeds_dim]), word_query
                                ), [self.total_sents, self.seq_len]
                            )
                        ), [self.total_sents, 1,self.seq_len]
                    ), embeds
                ), [self.total_sents, embeds_dim]
            )

        with tf.variable_scope('pcnn') as scope:
            x = tf.layers.conv1d(inputs=embeds, filters=self.params.cnn_dim, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            # 在final_process中计算出mask list，利用mask list做分段pooling
            mask_operator = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            mask = tf.nn.embedding_lookup(mask_operator, mask)
            x = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(x, 3), axis=1) - 100
            x = tf.reshape(x, [-1, self.params.cnn_dim * 3])
            # x=tf.layers.batch_normalization(x,training=True,center=True,scale=True,momentum=0.9) #loss爆炸时启用
            x = tf.nn.relu(x)
            x = tf.contrib.layers.dropout(x, keep_prob=self.params.dropout)
            cnn_out_dim = self.params.cnn_dim * 3

        #串联经word att得到的句子表示和经cnn得到的句子表示
        sent_repre=tf.concat([sent_repre,x],axis=1)
        de_out_dim=embeds_dim+cnn_out_dim

        with tf.variable_scope('sentence_attention') as scope:
            sentence_query = tf.get_variable('sentence_query', [de_out_dim, 1],
                                             initializer=tf.contrib.layers.xavier_initializer())

            def getSentenceAtt(num):
                num_sents = num[1] - num[0]
                bag_sents = sent_repre[num[0]:num[1]]

                sentence_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), sentence_query), [num_sents]))

                bag_repre_ = tf.reshape(
                    tf.matmul(
                        tf.reshape(sentence_att_weights, [1, num_sents]),
                        bag_sents
                    ), [de_out_dim]
                )
                return bag_repre_

            bag_repre = tf.map_fn(getSentenceAtt, self.sent_num, dtype=tf.float32)

        with tf.variable_scope('fully_connected_layer') as scope:
            w = tf.get_variable('w', [de_out_dim, self.num_class], initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=self.regularizer)
            b = tf.get_variable('b', initializer=np.zeros([self.num_class]).astype(np.float32),
                                regularizer=self.regularizer)
            nn_out = tf.nn.xw_plus_b(bag_repre, w, b)

        with tf.name_scope('Accuracy') as scope:
            prob = tf.nn.softmax(nn_out)
            y_pred = tf.argmax(prob, axis=1)
            y_actual = tf.argmax(self.input_y, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_actual), tf.float32))
        '''
        debugging command:
            res=debug_nn([])
        '''
        return nn_out, accuracy

    def add_loss(self, nn_out):
        with tf.name_scope('loss_op'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.input_y))
            if self.regularizer != None:
                loss += tf.contrib.layers.apply_regularization(self.regularizer,
                                                               tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss

    def add_optimizer(self, loss):
        with tf.name_scope('optimizer'):
            if self.params.optimizer == 'adam' and not self.params.restore:
                optimizer = tf.train.AdamOptimizer(self.params.lr)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.params.lr)
            train_op = optimizer.minimize(loss)
        return train_op

    def run_epoch(self, sess, data, epoch, shuffle=True):
        losses, accuracies = [], []
        bag_count = 0
        for step, batch in enumerate(self.getBatches(data, shuffle)):
            feed = self.create_feed_dict(batch)
            summary_str, loss, accuracy, _ = sess.run([self.merged_summary, self.loss, self.accuracy, self.train_op],
                                                      feed_dict=feed)
            losses.append(loss)
            accuracies.append(accuracy)

            bag_count += len(batch['sent_num'])

            if step % 10 == 0:
                self.logger.info('Epoch:{} Train Accuracy({}/{}):\t{:.5}\tLoss:{:.5}\t{}\tBest Train Acc:{:.5}'.format(epoch, bag_count,
                                                                                               len(self.data['train']),
                                                                                               np.mean(
                                                                                                   accuracies) * 100,
                                                                                               np.mean(losses),
                                                                                               self.params.name,
                                                                                               self.best_train_acc))
                self.summary_writer.add_summary(summary_str, epoch * len(self.data['train']) + bag_count)

        accuracy = np.mean(accuracies) * 100.0
        self.logger.info('Training loss:{},Accuracy:{}'.format(np.mean(losses), accuracy))
        return np.mean(losses), accuracy

    def predict(self, sess, data, wLabels=True, shuffle=False, label='Evaluating on test'):
        losses, accuracies, results, y_pred, y, logit_list, y_actual_hot = [], [], [], [], [], [], []
        bag_cnt = 0

        for step, batch in enumerate(self.getBatches(data, shuffle)):
            loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy],
                                              feed_dict=self.create_feed_dict(batch, dtype='test'))
            losses.append(loss)
            accuracies.append(accuracy)

            pred_index = logits.argmax(axis=1)
            logit_list += logits.tolist()
            y_actual_hot += self.getOneHot(batch["Y"], self.num_class).tolist()
            y_pred += pred_index.tolist()
            y += np.argmax(self.getOneHot(batch['Y'], self.num_class), 1).tolist()
            bag_cnt += len(batch['sent_num'])
            results.append(pred_index)

            if step % 100 == 0:
                self.logger.info('{} ({}/{}) :\t{:.5}\t{:.5}\t{}'.format(label, bag_cnt, len(self.data['test']),
                                                                         np.mean(accuracies) * 100, np.mean(losses),
                                                                         self.params.name))

        self.logger.info('Test Accuracy:{}'.format(accuracy))

        return np.mean(losses), results, np.mean(accuracies) * 100, y, y_pred, logit_list, y_actual_hot

    def calc_prec_recall_f1(self, y_actual, y_pred, none_id):
        pos_pred, pos_gt, true_pos = 0.0, 0.0, 0.0

        for i in range(len(y_actual)):
            if y_actual[i] != none_id:
                pos_gt += 1.0

        for i in range(len(y_pred)):
            if y_pred[i] != none_id:
                pos_pred += 1.0
                if y_pred[i] == y_actual[i]:
                    true_pos += 1.0
        prec = true_pos / (pos_pred + self.params.eps)
        recall = true_pos / (pos_gt + self.params.eps)
        f1 = 2 * prec * recall / (prec + recall + self.params.eps)

        return prec, recall, f1

    def getPNscore(self, data, label='P@N evaluation'):
        test_loss, test_pred, test_acc, y, y_pred, logit_list, y_hot = self.predict(sess, data, label)

        y_true = np.array([e[1:] for e in y_hot]).reshape((-1))
        y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))

        allprob = np.reshape(np.array(y_scores), (-1))
        allans = np.reshape(y_true, (-1))
        order = np.argsort(-allprob)

        def p_score(n):
            correct_num = 0
            for i in order[:n]:
                correct_num += 1.0 if (allans[i] == 1) else 0
            return correct_num / n

        return p_score(100), p_score(200), p_score(300)

    def fit(self, sess):
        self.summary_writer = tf.summary.FileWriter('tf_board/{}'.format(self.params.name), sess.graph)
        saver = tf.train.Saver()
        save_dir = 'checkpoints/{}/'.format(self.params.name)
        make_dir(save_dir)
        res_dir = 'results/{}/'.format(self.params.name)
        make_dir(res_dir)
        save_path = os.path.join(save_dir, 'best_model')

        # restore previous model
        if self.params.restore:
            saver.restore(sess, save_path)
        # train model
        if not self.params.only_eval:
            self.best_train_acc = 0.0
            not_best_count=0 # Stop training after several epochs without improvement.
            for epoch in range(self.params.max_epochs):
                train_loss, train_acc = self.run_epoch(sess, self.data['train'], epoch)
                self.logger.info(
                    '[Epoch {}]:Training Loss: {:.5},  Training Accuracy:{:.5}\n'.format(epoch, train_loss, train_acc))

                # store the model with least train loss
                if train_acc > self.best_train_acc:
                    self.best_train_acc = train_acc
                    saver.save(sess=sess, save_path=save_path)
                #     not_best_count=0
                # else:
                #     not_best_count+=1
                # if not_best_count>= 10:
                #     break


        # evaluation on test
        saver.restore(sess, save_path)
        test_loss, test_pred, test_acc, y, y_pred, logit_list, y_hot = self.predict(sess, self.data['test'])
        test_prec, test_recall, test_f1 = self.calc_prec_recall_f1(y, y_pred, 0)  # 0为na

        y_true = np.array([e[1:] for e in y_hot]).reshape((-1))
        y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))
        area_pr = average_precision_score(y_true, y_scores)

        self.logger.info(
            'Final results:Prec:{} | Recall:{} | F1:{} | Area:{}'.format(test_prec, test_recall, test_f1, area_pr))
        # store predictions
        pickle.dump({'logit list': logit_list, 'y_hot': y_hot},
                    open('results/{}/prec_recall.pkl'.format(self.params.name), 'wb'))

        # p@n evaluation
        # p@1
        one_100, one_200, one_300 = self.getPNscore(self.test_one, label='P@1 Evaluation')
        self.logger.info('TEST_ONE: P@100: {}, P@200: {}, P@300: {}'.format(one_100, one_200, one_300))
        one_avg = (one_100 + one_200 + one_300) / 3

        # P@2
        two_100, two_200, two_300 = self.getPNscore(self.test_two, label='P@2 Evaluation')
        self.logger.info('TEST_TWO: P@100: {}, P@200: {}, P@300: {}'.format(two_100, two_200, two_300))
        two_avg = (two_100 + two_200 + two_300) / 3

        # P@All
        all_100, all_200, all_300 = self.getPNscore(self.data['test'], label='P@All Evaluation')
        self.logger.info('TEST_THREE: P@100: {}, P@200: {}, P@300: {}'.format(all_100, all_200, all_300))
        all_avg = (all_100 + all_200 + all_300) / 3

        pprint({
            'one_100': one_100,
            'one_200': one_200,
            'one_300': one_300,
            'mean_one': one_avg,
            'two_100': two_100,
            'two_200': two_200,
            'two_300': two_300,
            'mean_two': two_avg,
            'all_100': all_100,
            'all_200': all_200,
            'all_300': all_300,
            'mean_all': all_avg,
        })


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="reconstruction of PCNN"
    )
    parser.add_argument('-name', dest='name', default='test_' + str(uuid.uuid4()), help='name of the run')
    parser.add_argument('-log_dir', dest='log_dir', default='./log/', help='log directory')
    parser.add_argument('-config_dir', dest='config_dir', default='./config/', help='config directorty')
    parser.add_argument('-data', dest='dataset', default='./riedel_processed.pkl', help='dataset to use')
    parser.add_argument('-gpu', dest='gpu', default='0', help='gpu to use')
    parser.add_argument('-cnn_dim', dest='cnn_dim', default=230, type=int, help='hidden state dimention of cnn')
    parser.add_argument('-pos_dim', dest='pos_dim', default=16, type=int, help='dimension of positional embedding')
    parser.add_argument('-drop', dest='dropout', default=0.8, type=float, help='Dropout for fully connected layer')
    parser.add_argument('-batch_size', dest='batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('-l2', dest='l2', default=0.001, type=float, help='l2 regularization')
    parser.add_argument('-embed_loc', dest='embed_loc', default='./glove/glove.6B.50d_word2vec.txt',
                        help='embed location')
    parser.add_argument('-embed_dim', dest='embed_dim', default=50, type=int, help='embed dimension')
    parser.add_argument('-optimizer', dest='optimizer', default='adam', help='optimizer for training')
    parser.add_argument('-restore', dest='restore', action='store_true', help='restore from the previous best model')
    parser.add_argument('-lr', dest='lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-only_eval', dest='only_eval', action='store_true',
                        help='Only evaluate pretrained model(skip training')
    parser.add_argument('-max_epochs', dest='max_epochs', default=2, type=int, help='Max epochs')
    parser.add_argument('-eps', dest='eps', default=0.00000001, type=float, help='value of epsilon')
    parser.add_argument('-chunk', dest='chunk_size', default=1000, type=int, help='chunk size')
    parser.add_argument('-seed', dest='seed', default=1234, type=int, help='seed for randomization')
    args = parser.parse_args()

    if not args.restore:
        args.name = args.name

    # set gpu
    set_gpu(args.gpu)

    # set seed
    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create model computional graph
    model = Model(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess)
