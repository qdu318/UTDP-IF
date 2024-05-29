# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import tensorflow as tf
import numpy as np
import logging
import os


def _Convolution2D(input, kernel_size, padding='SAME', is_relu=False):
    if is_relu:
        input = tf.nn.relu(input)
    with tf.variable_scope('conv_on_%s' % input.name[:-2]):
        weights = tf.get_variable(name='weight', shape=kernel_size, dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
    bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[kernel_size[3]]))
    conv = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding)
    convs = tf.nn.bias_add(conv, bias)
    return convs

def _Fusion(inputs):
    outputs = []
    for input in inputs:
        with tf.name_scope('fusion_%s' % input.name[:-2]):
            w = tf.Variable(tf.random_uniform(shape=[32, 32, 2], dtype=tf.float32), name='w')
            output = input * w
        outputs.append(output)
    return outputs[0] + outputs[1] + outputs[2]

def _FullCon(input, output_dim, is_relu=False):
    with tf.variable_scope('fc_%s' % input.name[:-2]):
        w = tf.get_variable(name='w', shape=[input.shape[1], output_dim],
                            initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[output_dim]), name='b')
    out = tf.nn.xw_plus_b(input, w, b)
    if is_relu:
        out = tf.nn.relu(out)
    return out


def gen_batch(x, y, index, batch_size=32):
    xc, xp, xt, ext = x[0], x[1], x[2], x[3]
    begin = index * batch_size
    end = begin + batch_size
    if (index + 1) * batch_size + batch_size > len(y):  # the last batch
        end = len(y)
    x_batch = [xc[begin:end], xp[begin:end], xt[begin:end], ext[begin:end], ]
    y_batch = y[begin:end]
    return x_batch, y_batch


class STResNet():
    def __init__(self,
                 learning_rate=0.0001,
                 epoches=50,
                 batch_size=32,
                 model_path='MODEL',
                 len_closeness=3,
                 len_period=1,
                 len_trend=1,
                 external_dim=28,
                 map_heigh=32,
                 map_width=32,
                 nb_flow=2,
                 nb_residual_unit=2):
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_path = model_path
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend
        self.external_dim = external_dim
        self.map_heigh = map_heigh
        self.map_width = map_width
        self.nb_flow = nb_flow
        self.nb_residual_unit = nb_residual_unit
        self.logger = logging.getLogger(__name__)
        self._build_placeholder()
        self._build_stresnet()

    def _build_placeholder(self):
        with tf.name_scope('model_inputs'):
            self.input_xc = tf.placeholder(dtype=tf.float32, shape=[None, self.map_heigh, self.map_width,
                                                                    self.nb_flow * self.len_closeness], name='input_xc')
            self.input_xp = tf.placeholder(dtype=tf.float32, shape=[None, self.map_heigh, self.map_width,
                                                                    self.nb_flow * self.len_period], name='input_xp')
            self.input_xt = tf.placeholder(dtype=tf.float32, shape=[None, self.map_heigh, self.map_width,
                                                                    self.nb_flow * self.len_trend], name='input_xt')
            self.input_ext = tf.placeholder(dtype=tf.float32, shape=[None, self.external_dim], name='input_external')
            self.output_y = tf.placeholder(dtype=tf.float32, shape=[None, self.map_heigh, self.map_width,
                                                                    2], name='output_y')

    def _build_stresnet(self):
        self.logger.info('### 构建网络...')  # 记录日志，指示正在构建网络
        with tf.name_scope('build_CPT'):  # 关闭，周期，趋势
            inputs = [self.input_xc, self.input_xp, self.input_xt]  # 输入包括关闭、周期、趋势
            outputs = []  # 存储每个输入对应的输出
            for input in inputs:
                conv1 = _Convolution2D(input, kernel_size=[3, 3, input.shape[3], 64])  # 第一个卷积操作
                # 使用 1x1 卷积将通道数调整为 2
                conv1 = tf.layers.conv2d(conv1, filters=2, kernel_size=1, padding='same')
                # LSTM 输入需要三维张量，形状为 (batch_size, timesteps, input_dim)
                # 将卷积的输出reshape为LSTM输入所需的形状
                lstm_input = tf.reshape(conv1, [-1, conv1.shape[1] * conv1.shape[2], 2])
                # 使用LSTM进行特征提取
                lstm_output= tf.keras.layers.LSTM(2, return_sequences=True)(lstm_input)
                # 将LSTM的输出reshape回四维张量
                lstm_output_reshaped = tf.reshape(lstm_output, [-1, conv1.shape[1], conv1.shape[2], 2])
                outputs.append(lstm_output_reshaped)

            if len(outputs) == 1:  # 如果只有一个输入，则直接取其输出作为主输出
                main_output = outputs[0]
            else:
                main_output = _Fusion(outputs)  # 否则，融合多个输入的输出
            self.logger.debug('### Fusion 操作后的形状:', main_output.shape)

        with tf.name_scope('build_E'):  # 外部数据
            if self.external_dim > 0:  # 如果存在外部数据
                embedding = _FullCon(self.input_ext, output_dim=10, is_relu=True)  # 全连接层，用于处理外部数据
                h1 = _FullCon(embedding, output_dim=self.map_heigh * self.map_width*2,is_relu=True)  # 全连接层
                external_output = tf.reshape(h1,shape=[-1, self.map_heigh, self.map_width, 2])  # reshape外部数据
                main_output += external_output  # 将外部数据加到主输出中
                self.logger.debug('### 添加外部数据后的形状:', main_output.shape)
        self.logits = tf.nn.tanh(main_output)  # 输出logits，通过tanh进行激活

        with tf.name_scope('loss'):  # 损失函数计算
            self.loss = tf.reduce_mean(tf.square(self.logits - self.output_y))  # 计算平方损失

    def evaluate(self, mmn, x, y):
        with tf.name_scope('mse'):
            _min = mmn._min
            _max = mmn._max
            predict = 0.5 * (self.logits + 1) * (_max - _min) + _min
            output_y = 0.5 * (self.output_y + 1) * (_max - _min) + _min
            square = tf.reduce_sum(tf.square(predict - output_y))

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(output_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(max_to_keep=3)

        with tf.Session() as sess:
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                saver.restore(sess, check_point)
            else:
                return
            n_chunk = len(y) // self.batch_size
            total_square = 0
            total_accuracy = 0

            for batch in range(n_chunk):
                x_batch, y_batch = gen_batch(x, y, batch, batch_size=self.batch_size)
                feed = {self.input_xc: x_batch[0],
                        self.input_xp: x_batch[1],
                        self.input_xt: x_batch[2],
                        self.input_ext: x_batch[3],
                        self.output_y: y_batch}
                batch_square, batch_accuracy = sess.run([square, accuracy], feed_dict=feed)
                total_square += batch_square
                total_accuracy += batch_accuracy

            rmse = np.sqrt(total_square / (len(y) * self.map_heigh * self.map_width * self.nb_flow))
            avg_accuracy = total_accuracy / n_chunk
            print('### RMSE:', rmse)
            print('### Accuracy:', avg_accuracy)

    def train(self, x, y):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        with tf.device('/gpu:0'):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.logger.info('### Training...')
        saver = tf.train.Saver(max_to_keep=3)
        model_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
            self.len_closeness, self.len_period, self.len_trend, self.nb_residual_unit, self.learning_rate)
        with tf.Session() as sess:
            self.logger.debug('### Initializing...')
            sess.run(tf.global_variables_initializer())
            start_epoch = 0
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                try:
                    saver.restore(sess, check_point)
                    start_epoch += int(check_point.split('-')[-1])
                    self.logger.info("### Loading exist model <{}> successfully...".format(check_point))
                except Exception as e:
                    # 增加日志记录，指示恢复检查点失败
                    self.logger.warning("Failed to load checkpoint. Starting from scratch. Error: {}".format(e))
                    # 删除旧的检查点文件
                    for file in os.listdir(self.model_path):
                        file_path = os.path.join(self.model_path, file)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)  # 标记：删除旧的检查点文件
            total_loss = 0
            try:
                for epoch in range(start_epoch, self.epoches):
                    n_chunk = len(y) // self.batch_size
                    ave_loss = total_loss / n_chunk
                    total_loss = 0
                    for batch in range(n_chunk):
                        x_batch, y_batch = gen_batch(x, y, batch, batch_size=self.batch_size)
                        feed = {self.input_xc: x_batch[0],
                                self.input_xp: x_batch[1],
                                self.input_xt: x_batch[2],
                                self.input_ext: x_batch[3],
                                self.output_y: y_batch}
                        loss, _ = sess.run([self.loss, train_op], feed_dict=feed)
                        total_loss += loss
                        if batch % 50 == 0:
                            self.logger.info(
                                '### Epoch:%d, last epoch loss ave:%.5f batch:%d, current epoch loss:%.5f' % (
                                    epoch, ave_loss, batch, loss))
                    if epoch % 3 == 0:
                        self.logger.info('### Saving model...')
                        saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch)
            except KeyboardInterrupt:
                self.logger.warning("KeyboardInterrupt saving...")
                saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch - 1)



