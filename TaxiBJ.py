from __future__ import print_function
import os, sys
from datetime import datetime

sys.path.append('../../')
import time
import pickle
from copy import copy
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from data.TaxiBJ.preprocessing.STMatrix import STMatrix
from data.TaxiBJ.preprocessing.timestamp import timestamp2vec
from data.TaxiBJ.preprocessing.MaxMinNormalization import MinMaxNormalization

# parameters
DATAPATH = os.path.dirname(os.path.abspath(__file__))
CACHEPATH = os.path.join(DATAPATH, 'CACHE')

def load_holiday_with_feature_selection(timeslots, fname=os.path.join(DATAPATH, 'BJ_Holiday.txt')):
    # 读取假期数据
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])

    # 从时间戳生成特征（例如，星期几，月份等）
    features = generate_features_from_timestamps(timeslots)  # 需要实现此函数

    # 将假期转换为二进制标签
    labels = np.array([1 if slot[:8] in holidays else 0 for slot in timeslots])

    # 使用随机森林进行特征选择
    clf = RandomForestClassifier()
    clf.fit(features, labels)
    important_features = clf.feature_importances_ > 0.05  # 示例重要性阈值

    # 选择重要特征
    selected_features = features[:, important_features]

    # 使用PCA进行降维
    pca = PCA(n_components=2)  # 示例组件数
    reduced_features = pca.fit_transform(selected_features)

    # 为降维后的特征生成二进制标签
    reduced_labels = np.array([1 if slot[:8] in holidays else 0 for slot in timeslots])

    return reduced_labels[:, None]  # 返回为2D数组


def generate_features_from_timestamps(timeslots):
    features = []
    for slot in timeslots:
        # 将时间戳转换为 datetime 对象
        year = int(slot[:4])
        month = int(slot[4:6])
        day = int(slot[6:8])
        hour = int(slot[8:10])
        dt = datetime.datetime(year, month, day, hour)

        # 提取特征
        weekday = dt.weekday()  # 星期几，0代表星期一，6代表星期日
        month = dt.month  # 月份
        hour = dt.hour  # 小时

        # 可以根据需要添加更多特征，例如是否是周末
        is_weekend = 1 if weekday >= 5 else 0

        # 将特征添加到列表中
        features.append([weekday, month, hour, is_weekend])

    return np.array(features)


def load_meteorol_with_feature_selection(timeslots, fname=os.path.join(DATAPATH, 'BJ_Meteorology.h5')):
    # 读取气象数据
    f = h5py.File(fname, 'r')
    Timeslot = f['date'][:]
    WindSpeed = f['WindSpeed'][:]
    Weather = f['Weather'][:]
    Temperature = f['Temperature'][:]
    f.close()

    # 构建时间片到索引的映射
    M = dict()
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    # 收集气象数据
    WS = []  # 风速
    WR = []  # 天气
    TE = []  # 温度
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1  # 使用前一个时间片的数据
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 归一化
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("气象数据形状: ", WS.shape, WR.shape, TE.shape)

    # 合并所有气象特征
    features = np.hstack([WR, WS[:, None], TE[:, None]])

    # 使用随机森林进行特征选择
    clf = RandomForestRegressor()
    clf.fit(features, np.ones(len(features)))  # 这里使用虚拟标签，因为我们只关心特征重要性
    important_features = clf.feature_importances_ > 0.05  # 示例重要性阈值

    # 选择重要特征
    selected_features = features[:, important_features]

    # 使用PCA进行降维
    pca = PCA(n_components=2)  # 示例组件数
    reduced_features = pca.fit_transform(selected_features)

    return reduced_features  # 返回降维后的气象数据

def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'][:]
    timestamps = f['date'][:]
    f.close()
    return data, timestamps


def stat(fname):

    def get_nb_timeslot(f):
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        time_s_str, time_e_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, time_s_str, time_e_str

    with h5py.File(fname) as f:
        nb_timeslot, time_s_str, time_e_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = f['data'][:].max()
        mmin = f['data'][:].min()
        stat = '=' * 10 + 'stat' + '=' * 10 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, time_s_str, time_e_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 10 + 'stat' + '=' * 10
        print(stat)


def remove_incomplete_days(data, timestamps, T=48):
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def load_dataset(T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
                 len_test=None, preprocess_name='preprocessing.pkl',
                 meta_data=True, meteorol_data=True, holiday_data=True):
    assert (len_closeness + len_period + len_trend > 0)
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
            DATAPATH, 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)

    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]
    fpkl = open(os.path.join(DATAPATH, CACHEPATH, preprocess_name), 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)  # 保存特征缩放模型[-1,1]
    fpkl.close()
    print(timestamps_all[0][:10])
    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y  # [ b'2013102232', b'2013102233', b'2013102234', b'2013102235',......]
    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)  # array: [?,8]
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday_with_feature_selection(timestamps_Y)
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol_with_feature_selection(timestamps_Y)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)  # shape = [15072,6,32,32]
    XP = np.vstack(XP)  # shape = [15072,2,32,32]
    XT = np.vstack(XT)  # shape = [15072,2,32,32]
    Y = np.vstack(Y)  # shape = [15072,2,32,32]

    XC=np.transpose(XC,[0,2,3,1])
    XP=np.transpose(XP,[0,2,3,1])
    XT=np.transpose(XT,[0,2,3,1])
    Y=np.transpose(Y,[0,2,3,1])

    print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[:-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('XC_train shape:', XC_train.shape, Y_train.shape, 'XC_test shape: ', XC_test.shape, Y_test.shape)
    #
    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)

    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()


def read_cache(fname):
    mmn = pickle.load(open(os.path.join(DATAPATH, CACHEPATH, 'preprocessing.pkl'), 'rb'))
    f = h5py.File(fname, 'r')
    num = int(f['num'][()])
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(f['X_train_%i' % i][:])
        X_test.append(f['X_test_%i' % i][:])
    Y_train = f['Y_train'][:]
    Y_test = f['Y_test'][:]
    external_dim = f['external_dim'][()]
    timestamp_train = f['T_train'][:]
    timestamp_test = f['T_test'][:]
    f.close()
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


def load_data(len_closeness, len_period, len_trend, len_test, meta_data=True, meteorol_data=True, holiday_data=True):
    fname = os.path.join(DATAPATH, CACHEPATH, 'TaxiBJ_C{}_P{}_T{}.h5'.format(len_closeness, len_period, len_trend))
    if os.path.exists(fname):
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
            fname)
        print("load %s successfully" % fname)
    else:
        if os.path.isdir(CACHEPATH) is False:
            os.mkdir(CACHEPATH)
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
            load_dataset(len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                         len_test=len_test, meta_data=True, meteorol_data=True, holiday_data=True)
        cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test)
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
        load_data(len_closeness=3, len_period=1, len_trend=1, len_test=28 * 48)
