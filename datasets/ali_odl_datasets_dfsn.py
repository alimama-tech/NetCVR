
import shutil
import pickle
import pandas as pd
import argparse
import copy
import os
import numpy as np
import pickle
from tqdm import tqdm
import hashlib
from pandas.util import hash_pandas_object
SECONDS_A_DAY = 60*60*24
SECONDS_AN_HOUR = 60*60
SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY*5
num_bin_size = ()
cate_bin_size = (8,327680,655360,655360,8192,655360,16,32,256,81920,163840,8,512,128,256,1024,128,16,8,8,16,16)
def get_data_df(args):
    print("Loading data from ", args.data_path)
    df = pd.read_csv(args.data_path, sep="\t", header=None)
    print(df.head())
    print("preprocessing data from ",args.data_path)
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()
    refund_ts = df[df.columns[2]].fillna(-1).to_numpy()
    df = df[df.columns[3:]]
    df.columns = [str(i) for i in range(22)]
    df.reset_index(drop=True)
    return df, click_ts, pay_ts, refund_ts


class DataDF(object):

    def __init__(self, features, click_ts, pay_ts, refund_ts, sample_ts=None, pay_labels=None,net_pay_labels =None,refund_labels=None,stream_pay_labels=None,stream_net_pay_labels=None,
                 stream_pay_mask =None,delay_pay_labels_afterPay=None, delay_pay_label_afterRefund = None,delay_refund_label_afterRefund=None,inw_pay_labels_afterPay= None, inw_pay_labels_afterRefund=None, pay_attr_window = None, refund_attr_window = None):
        """
        初始化方法。
        Args:
            features (pandas.DataFrame): 特征数据，包含数值特征或ID特征。
            click_ts (numpy.ndarray): 点击时间戳数组。
            pay_ts (numpy.ndarray): 转化时间戳数组。
            refund_ts (numpy.ndarray): 退款时间戳数组。
            sample_ts (numpy.ndarray, optional): 采样时间戳数组，默认为 None。如果为 None，则使用点击时间戳作为采样时间戳。
            pay_labels (numpy.ndarray, optional): 真实的转化标签数组，默认为 None。如果为 None，则根据支付时间和点击时间生成转化标签。
            net_pay_labels (numpy.ndarray, optional): 真实的净转化标签数组（考虑转化和退款），默认为 None。如果为 None，则根据支付时间、退款时间和点击时间生成净转化标签。
            refund_labels (numpy.ndarray, optional): 真实的退款标签数组，默认为 None。如果为 None，则根据支付时间、退款时间和点击时间生成退款标签。
            stream_pay_labels (numpy.ndarray, optional): 流转化标签数组(主模型)，默认为 None。如果为 None，则使用转化标签作为流转化标签。
            stream_net_pay_labels (numpy.ndarray, optional): 流净转化标签数组(主模型)，默认为 None。如果为 None，则使用净转化标签作为流净转化标签。
            stream_pay_mask (numpy.ndarray, optional): 流转化标签掩码数组，默认为 None。如果为 None，则使用转化标签作为流转化标签掩码.用来区分是不是退款相关的标签
            delay_pay_labels_afterPay (numpy.ndarray, optional): 延迟转化的转化标签数组，默认为 None。如果为 None，则使用转化标签作为延迟转化的转化标签。
            delay_pay_label_afterRefund (numpy.ndarray, optional): 延迟退款的转化标签数组，默认为 None。如果为 None，则使用净转化标签作为延迟退款的转化标签。
            delay_refund_label_afterRefund (numpy.ndarray, optional): 延迟退款的退款标签数组，默认为 None。如果为 None，则使用退款标签作为延迟退款的退款标签。
            inw_pay_labels_afterPay (numpy.ndarray, optional): 在转化窗口内转化的转化标签数组，默认为 None。如果为 None，则使用转化标签作为在转化窗口内转化的转化标签。
            inw_pay_labels_afterRefund (numpy.ndarray, optional): 在退款窗口内考虑退款后的转化标签数组，默认为 None。如果为 None，则使用净转化标签作为在退款窗口内考虑退款后的转化标签。
            pay_attr_window (int, optional): 归因窗口（支付时间窗口），默认为 None。用于生成转化标签和净转化标签。
            refund_attr_window (int, optional): 归因窗口（退款时间窗口），默认为 None。用于生成退款标签。

        Returns:
            None

        """
        self.features = features.copy(deep=True)

        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.refund_ts = copy.deepcopy(refund_ts)
        self.delay_pay_labels_afterPay = delay_pay_labels_afterPay
        self.delay_pay_label_afterRefund = delay_pay_label_afterRefund
        self.inw_pay_labels_afterPay = inw_pay_labels_afterPay
        self.delay_refund_label_afterRefund = delay_refund_label_afterRefund
        self.inw_pay_labels_afterRefund = inw_pay_labels_afterRefund
        self.stream_pay_labels = stream_pay_labels
        self.stream_net_pay_labels = stream_net_pay_labels

        self.stream_pay_mask = stream_pay_mask
        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        if pay_labels is not None:
            self.pay_labels = copy.deepcopy(pay_labels)
        else:
            if pay_attr_window is not None:
                self.pay_labels = (np.logical_and(pay_ts > 0, pay_ts - click_ts < pay_attr_window)).astype(np.int32)
            else:
                self.pay_labels = (pay_ts > 0).astype(np.int32)
        
        if net_pay_labels is not None:
            self.net_pay_labels = copy.deepcopy(net_pay_labels)
        else:
            if pay_attr_window is not None and refund_attr_window is not None:
                self.net_pay_labels = (
                    np.logical_and(
                        pay_ts > 0,
                        np.logical_and(
                            pay_ts - click_ts < pay_attr_window,
                            np.logical_or(
                                refund_ts < 0,
                                refund_ts - pay_ts >= refund_attr_window
                            )
                        )
                    )
                ).astype(np.int32)
            else:
                self.net_pay_labels = (np.logical_and(pay_ts > 0, refund_ts < 0)).astype(np.int32)

        if refund_labels is not None:
            self.refund_labels = copy.deepcopy(refund_labels)
        else:
            if pay_attr_window is not None and refund_attr_window is not None:
                self.refund_labels = (
                    np.logical_and(
                        pay_ts > 0,
                            np.logical_and(
                                pay_ts - click_ts < pay_attr_window,
                                np.logical_and(
                                    refund_ts > 0,
                                    refund_ts - pay_ts < refund_attr_window
                                )
                        )
                    )
                ).astype(np.int32)
            else:
                self.refund_labels = (np.logical_and(pay_ts > 0, refund_ts > 0)).astype(np.int32)


        
        if self.delay_pay_labels_afterPay is None:
            self.delay_pay_labels_afterPay = self.pay_labels
        if self.inw_pay_labels_afterPay is None:
            self.inw_pay_labels_afterPay = self.pay_labels


        if self.delay_pay_label_afterRefund is None:
            self.delay_pay_label_afterRefund = self.net_pay_labels
        if self.delay_refund_label_afterRefund is None:
            self.delay_refund_label_afterRefund = self.refund_labels
        if self.inw_pay_labels_afterRefund is None:
            self.inw_pay_labels_afterRefund = self.net_pay_labels

        if self.stream_pay_labels is None:
            self.stream_pay_labels = self.pay_labels
        if self.stream_net_pay_labels is None:
            self.stream_net_pay_labels = self.net_pay_labels


        
        if self.stream_pay_mask is None:
            self.stream_pay_mask = np.ones_like(self.pay_labels)

        self.pay_attr_window = pay_attr_window
        self.refund_attr_window = refund_attr_window

    def sub_days(self,start_day,end_day):
        """
        提取指定日期范围内的数据。
        Args:
            start_day (int): 起始日期。
            end_day (int): 结束日期。
        Returns:
            DataDF: 返回一个新的DataDF对象，其中包含了指定日期范围内的数据。
        """
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY 
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)

        return DataDF(features = self.features.iloc[mask],
                click_ts = self.click_ts[mask],
                pay_ts = self.pay_ts[mask],
                refund_ts = self.refund_ts[mask],
                sample_ts = self.sample_ts[mask],
                pay_labels = self.pay_labels[mask],
                net_pay_labels = self.net_pay_labels[mask],
                refund_labels = self.refund_labels[mask],
                delay_pay_labels_afterPay = self.delay_pay_labels_afterPay[mask],
                delay_pay_label_afterRefund = self.delay_pay_label_afterRefund[mask],
                delay_refund_label_afterRefund = self.delay_refund_label_afterRefund[mask],
                inw_pay_labels_afterPay = self.inw_pay_labels_afterPay[mask],
                inw_pay_labels_afterRefund = self.inw_pay_labels_afterRefund[mask],
                stream_pay_labels = self.stream_pay_labels[mask],
                stream_net_pay_labels = self.stream_net_pay_labels[mask],
                stream_pay_mask= self.stream_pay_mask[mask],
                pay_attr_window=self.pay_attr_window,
                refund_attr_window=self.refund_attr_window
                )

    def sub_days_v2(self, start_day, end_day, pay_wait_window):
        """
        根据给定的 wait_window,attr_window提取指定日期范围内的数据。
        V2版本我们开始考虑实现转化延迟标签(不考虑退款)
        Args:
            start_day (int): 开始日期。
            end_day (int): 结束日期。
            wait_window (int): 截断大小。
        """
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,self.sample_ts < end_ts)
        if self.pay_attr_window is not None:
            diff = self.pay_ts - self.click_ts
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff > pay_wait_window, diff < self.pay_attr_window))
        else:
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > pay_wait_window)
        
        delay_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        delay_pay_labels_afterPay[~delay_mask_afterPay] = 0

        return DataDF(features = self.features.iloc[mask],
                      click_ts = self.click_ts[mask],
                      pay_ts = self.pay_ts[mask],
                      sample_ts = self.sample_ts[mask],
                      refund_ts = self.refund_ts[mask],
                      pay_labels=self.pay_labels[mask],
                      net_pay_labels = self.net_pay_labels[mask],
                      refund_labels = self.refund_labels[mask],
                      delay_pay_labels_afterPay = delay_pay_labels_afterPay[mask],
                      delay_pay_label_afterRefund = self.delay_pay_label_afterRefund[mask],
                      delay_refund_label_afterRefund = self.delay_refund_label_afterRefund[mask],
                      inw_pay_labels_afterPay = self.inw_pay_labels_afterPay[mask],
                      inw_pay_labels_afterRefund = self.inw_pay_labels_afterRefund[mask],
                      stream_pay_labels = self.stream_pay_labels[mask],
                      stream_net_pay_labels = self.stream_net_pay_labels[mask],
                      stream_pay_mask= self.stream_pay_mask[mask],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)
    
    def sub_days_v3(self, start_day, end_day, pay_wait_window):
        """
        根据给定的 wait_window,attr_window提取指定日期范围内的数据。
        V3版本我们开始考虑实现转化延迟标签,及时转化样本(不考虑退款)
        Args:
            start_day (int): 开始日期。
            end_day (int): 结束日期。
            wait_window (int): 截断大小。
        """
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,self.sample_ts < end_ts)
        if self.pay_attr_window is not None:
            diff = self.pay_ts - self.click_ts
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff > pay_wait_window, diff < self.pay_attr_window))
            inw_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff <= pay_wait_window, diff < self.pay_attr_window))
        else:
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > pay_wait_window)
            inw_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= pay_wait_window)
       
        delay_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        delay_pay_labels_afterPay[~delay_mask_afterPay] = 0

        inw_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        inw_pay_labels_afterPay[~inw_mask_afterPay] = 0

        return DataDF(features = self.features.iloc[mask],
                      click_ts = self.click_ts[mask],
                      pay_ts = self.pay_ts[mask],
                      sample_ts = self.sample_ts[mask],
                      refund_ts = self.refund_ts[mask],
                      pay_labels=self.pay_labels[mask],
                      net_pay_labels = self.net_pay_labels[mask],
                      refund_labels = self.refund_labels[mask],
                      delay_pay_labels_afterPay = delay_pay_labels_afterPay[mask],
                      delay_pay_label_afterRefund = self.delay_pay_label_afterRefund[mask],
                      delay_refund_label_afterRefund = self.delay_refund_label_afterRefund[mask],
                      inw_pay_labels_afterPay = inw_pay_labels_afterPay[mask],
                      inw_pay_labels_afterRefund = self.inw_pay_labels_afterRefund[mask],
                      stream_pay_labels = self.stream_pay_labels[mask],
                      stream_net_pay_labels = self.stream_net_pay_labels[mask],
                      stream_pay_mask= self.stream_pay_mask[mask],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def sub_days_v4(self, start_day, end_day, pay_wait_window,refund_wait_window):
        """
        根据给定的 wait_window,attr_window提取指定日期范围内的数据。
        V4版本我们开始考虑实现转化延迟标签,及时转化样本(不考虑退款),延迟退款样本(考虑退款),及时退款样本(考虑退款)
        Args:
            start_day (int): 开始日期。
            end_day (int): 结束日期。
            wait_window (int): 截断大小。
        """
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,self.sample_ts < end_ts)
        if self.pay_attr_window is not None:
            diff = self.pay_ts - self.click_ts
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff > pay_wait_window, diff < self.pay_attr_window))
            inw_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff <= pay_wait_window, diff < self.pay_attr_window))
        else:
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > pay_wait_window)
            inw_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= pay_wait_window)

        if self.refund_attr_window is not None:
            diff = self.refund_ts - self.pay_ts
            delay_mask_afterRefund = np.logical_and(self.pay_ts > 0, np.logical_and(self.refund_ts > 0, np.logical_and(diff > refund_wait_window, diff < self.refund_attr_window)))
            inw_mask_afterRefund = np.logical_and(self.pay_ts > 0, np.logical_and(self.refund_ts > 0, np.logical_and(diff <= refund_wait_window, diff < self.refund_attr_window)))
        else:
            delay_mask_afterRefund = np.logical_and(self.pay_ts > 0, np.logical_and(self.refund_ts > 0, self.refund_ts - self.pay_ts > refund_wait_window))
            inw_mask_afterRefund = np.logical_and(self.pay_ts > 0, np.logical_and(self.refund_ts > 0, self.refund_ts - self.pay_ts <= refund_wait_window))
       
        delay_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        delay_pay_labels_afterPay[~delay_mask_afterPay] = 0

        inw_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        inw_pay_labels_afterPay[~inw_mask_afterPay] = 0

        delay_refund_labels_afterRefund = copy.deepcopy(self.refund_labels)
        delay_refund_labels_afterRefund[~delay_mask_afterRefund] = 0


        return DataDF(features = self.features.iloc[mask],
                      click_ts = self.click_ts[mask],
                      pay_ts = self.pay_ts[mask],
                      sample_ts = self.sample_ts[mask],
                      refund_ts = self.refund_ts[mask],
                      pay_labels=self.pay_labels[mask],
                      net_pay_labels = self.net_pay_labels[mask],
                      refund_labels = self.refund_labels[mask],
                      delay_pay_labels_afterPay = delay_pay_labels_afterPay[mask],
                      delay_pay_label_afterRefund = self.delay_pay_label_afterRefund[mask],
                      delay_refund_label_afterRefund = delay_refund_labels_afterRefund[mask],
                      inw_pay_labels_afterPay = inw_pay_labels_afterPay[mask],
                      inw_pay_labels_afterRefund = self.inw_pay_labels_afterRefund[mask],
                      stream_pay_labels = self.stream_pay_labels[mask],
                      stream_net_pay_labels = self.stream_net_pay_labels[mask],
                      stream_pay_mask= self.stream_pay_mask[mask],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_dfsn_main_duplicate_samples(self, pay_wait_window):
        """
        在给定归因窗口和观测窗口内，生成带有延迟标签的样本，并返回DataDF对象。注意此时我们不考虑退款
        dfsn 主塔有等待窗口，且回补延迟正样本

        Args:
            pay_wait_window (int): 观测窗口的大小，表示从点击到支付的最大等待时间

        Returns:
            DataDF: 包含新生成的带有延迟标签的样本的DataDF对象。

        """
        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        delay_pay_mask = np.logical_and(self.pay_labels , (self.pay_ts - self.click_ts) > pay_wait_window)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
    
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        ])
        sample_ts = np.concatenate([
            self.pay_ts[inw_pay_mask],
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask]
        ])
        click_ts = np.concatenate([self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask], \
            self.click_ts[delay_pay_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask], \
            self.pay_ts[delay_pay_mask]], axis=0)
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], \
                        self.refund_ts[delay_pay_mask]], axis=0)
    
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),
        np.zeros((np.sum(~inw_pay_mask),)),
        pay_labels[delay_pay_mask],
        ], axis=0)

        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask]                             
        ], axis=0)

        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask]          

        ], axis=0)

        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask]          
        ], axis=0)

        idx = list(range(new_features.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(features=new_features.iloc[idx],
                      click_ts=click_ts[idx],
                      pay_ts=pay_ts[idx],
                      sample_ts=sample_ts[idx],
                      refund_ts=refund_ts[idx],
                      pay_labels=pay_labels[idx],
                      net_pay_labels=net_pay_labels[idx],
                      refund_labels=refund_labels[idx],
                      stream_pay_labels=stream_pay_labels[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)


    def add_dfsn_main_duplicate_samples1(self, pay_wait_window):
        """
        在给定归因窗口和观测窗口内，生成带有延迟标签的样本，并返回DataDF对象。注意此时我们不考虑退款
        dfsn 主塔有等待窗口，且回补延迟正样本

        Args:
            pay_wait_window (int): 观测窗口的大小，表示从点击到支付的最大等待时间

        Returns:
            DataDF: 包含新生成的带有延迟标签的样本的DataDF对象。

        """
        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        delay_pay_mask = np.logical_and(self.pay_labels, (self.pay_ts - self.click_ts) > pay_wait_window)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
    
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        ])
        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask]
        ])
        click_ts = np.concatenate([self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask], \
            self.click_ts[delay_pay_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask], \
            self.pay_ts[delay_pay_mask]], axis=0)
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], \
                        self.refund_ts[delay_pay_mask]], axis=0)
    
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),
        np.zeros((np.sum(~inw_pay_mask),)),
        pay_labels[delay_pay_mask],
        ], axis=0)

        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask]                             
        ], axis=0)

        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask]          

        ], axis=0)

        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask]          
        ], axis=0)

        idx = list(range(new_features.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(features=new_features.iloc[idx],
                      click_ts=click_ts[idx],
                      pay_ts=pay_ts[idx],
                      sample_ts=sample_ts[idx],
                      refund_ts=refund_ts[idx],
                      pay_labels=pay_labels[idx],
                      net_pay_labels=net_pay_labels[idx],
                      refund_labels=refund_labels[idx],
                      stream_pay_labels=stream_pay_labels[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_dfsn_feature_duplicate_samples(self, pay_wait_window):
        """
        在给定归因窗口和观测窗口内，生成带有延迟标签的样本，并返回DataDF对象。注意此时我们不考虑退款
        dfsn feature塔有等待窗口内的样本，但是没有延迟样本回补

        Args:
            pay_wait_window (int): 观测窗口的大小，表示从点击到支付的最大等待时间

        Returns:
            DataDF: 包含新生成的带有延迟标签的样本的DataDF对象。

        """
        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        delay_pay_mask = np.logical_and(self.pay_labels, (self.pay_ts - self.click_ts) > pay_wait_window)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
    
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        ])
        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
        ])
        click_ts = np.concatenate([self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask]], axis=0)
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask]], axis=0)
    
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),
        np.zeros((np.sum(~inw_pay_mask),))
        ], axis=0)

        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                                           
        ], axis=0)

        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],           
        ], axis=0)

        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],              
        ], axis=0)

        idx = list(range(new_features.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(features=new_features.iloc[idx],
                      click_ts=click_ts[idx],
                      pay_ts=pay_ts[idx],
                      sample_ts=sample_ts[idx],
                      refund_ts=refund_ts[idx],
                      pay_labels=pay_labels[idx],
                      net_pay_labels=net_pay_labels[idx],
                      refund_labels=refund_labels[idx],
                      stream_pay_labels=stream_pay_labels[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_dfsn_unbias_duplicate_samples(self):
        """
        添加假负样本以处理延迟转化。
        这里dfsn的无偏塔参考fnw的实现就好
        Args:
            无
        Returns:
            DataDF: 包含假负样本的数据对象。
        """

        pos_mask = np.logical_and(self.pay_ts>0 , self.pay_labels)

        new_features = pd.concat([
        self.features.copy(deep=True),
        self.features.iloc[pos_mask].copy(deep=True)
        ])

        sample_ts = np.concatenate([
            self.click_ts,
            self.pay_ts[pos_mask]
        ])
        click_ts = np.concatenate([
            self.click_ts, self.click_ts[pos_mask]
        ])
        pay_ts = np.concatenate([
            self.pay_ts, self.pay_ts[pos_mask]
        ])
        refund_ts = np.concatenate([
            self.refund_ts, self.refund_ts[pos_mask]
        ])

        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)
        stream_pay_labels = copy.deepcopy(self.pay_labels)
        stream_pay_labels[pos_mask] = 0

        stream_pay_labels = np.concatenate([np.zeros_like(stream_pay_labels), np.ones((np.sum(pos_mask),))],axis=0)
        pay_labels = np.concatenate([pay_labels, pay_labels[pos_mask]],axis=0)
        net_pay_labels = np.concatenate([net_pay_labels, net_pay_labels[pos_mask]],axis=0)
        refund_labels = np.concatenate([refund_labels, refund_labels[pos_mask]],axis=0)

        idx = list(range(new_features.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])

        return DataDF(features=new_features.iloc[idx],
                      click_ts=click_ts[idx],
                      pay_ts=pay_ts[idx],
                      sample_ts=sample_ts[idx],
                      refund_ts=refund_ts[idx],
                      pay_labels=pay_labels[idx],
                      net_pay_labels=net_pay_labels[idx],
                      refund_labels=refund_labels[idx],
                      stream_pay_labels=stream_pay_labels[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)


    def shuffle(self):
        """
        打乱数据顺序。

        Args:
            无

        Returns:
            DataDF: 打乱顺序后的数据。

        """
        idx = list(range(self.features.shape[0]))
        np.random.shuffle(idx)
        return DataDF(features = self.features.iloc[idx],
                      click_ts = self.click_ts[idx],
                      pay_ts = self.pay_ts[idx],
                      refund_ts = self.refund_ts[idx],
                      sample_ts = self.sample_ts[idx],
                      pay_labels = self.pay_labels[idx],
                      net_pay_labels = self.net_pay_labels[idx], 
                      refund_labels= self.refund_labels[idx],
                      delay_pay_labels_afterPay = self.delay_pay_labels_afterPay[idx],
                      inw_pay_labels_afterPay = self.inw_pay_labels_afterPay[idx],
                      delay_pay_label_afterRefund=self.delay_pay_label_afterRefund[idx],
                      delay_refund_label_afterRefund=self.delay_refund_label_afterRefund[idx],
                      inw_pay_labels_afterRefund=self.inw_pay_labels_afterRefund[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

def get_ali_dataset_pretrain(args):
    np.random.seed(args.seed)
    dataset_name = args.dataset_name
    mode = args.mode
    print("Loading data from {}".format(args.data_cache_path))
    cache_path = os.path.join(args.data_cache_path, f"{dataset_name}_{mode}.pkl")
    if os.path.isfile(cache_path):
        print("cache_path {} exists.".format(cache_path))
        print("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        train_data = data["train"]
        test_data = data["test"]
    else:
        print("building datasets")
        df,click_ts,pay_ts,refund_ts = get_data_df(args)
        pay_wait_window = args.pay_wait_window * SECONDS_A_DAY
        refund_wait_window = args.refund_wait_window * SECONDS_A_DAY
        pay_attr_window = args.pay_attr_window * SECONDS_A_DAY
        refund_attr_window = args.refund_attr_window * SECONDS_A_DAY
        data_src = DataDF(features = df, click_ts=click_ts, pay_ts=pay_ts, refund_ts=refund_ts,pay_attr_window=pay_attr_window,refund_attr_window=refund_attr_window)

        print("splitting into train and test sets")
        if mode == "dfsn_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        else:
            raise ValueError(f"Unknown mode {mode}")
        print("writing data to cache file")
        
    print("data loaded successfully")
    print("====== TRAIN SET ======")
    print(f"Total samples                : {len(train_data.pay_labels):,}")
    print(f"Positive pay labels         : {sum(train_data.pay_labels):,}")
    print(f"Positive net pay labels   : {sum(train_data.net_pay_labels):,}")
    print(f"Positive refund labels      : {sum(train_data.refund_labels):,}")
    print(f"Positive delay pay labels   : {sum(train_data.delay_pay_labels_afterPay):,}")
    print(f"Positive delay refund labels   : {sum(train_data.delay_refund_label_afterRefund):,}")
    print("\n====== TEST SET ======")
    if hasattr(test_data, 'pay_labels'):
        print(f"Total samples                : {len(test_data.pay_labels):,}")
        print(f"Positive pay labels         : {sum(test_data.pay_labels):,}")
        print(f"Positive net pay labels   : {sum(test_data.net_pay_labels):,}")
        print(f"Positive refund labels      : {sum(test_data.refund_labels):,}")
        print(f"Positive delay pay labels   : {sum(test_data.delay_pay_labels_afterPay):,}")
        print(f"Positive delay refund labels   : {sum(test_data.delay_refund_label_afterRefund):,}")
    else:
        print(f"Total samples                : {len(test_data[0]['pay_labels']):,}")
        print(f"Positive pay labels         : {sum(test_data[0]['pay_labels']):,}")
        print(f"Positive net pay labels   : {sum(test_data[0]['net_pay_labels']):,}")
        print(f"Positive refund labels      : {sum(test_data[0]['refund_labels']):,}")
        print(f"Positive delay pay labels   : {sum(test_data[0]['delay_pay_labels_afterPay']):,}")
        print(f"Positive delay refund labels   : {sum(test_data[0]['delay_refund_label_afterRefund']):,}")


    return {
        "train": {
            "features": train_data.features,
            "click_ts": train_data.click_ts,
            "pay_ts": train_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "refund_ts": train_data.refund_ts,
            "pay_labels": train_data.pay_labels,
            "net_pay_labels": train_data.net_pay_labels,
            "refund_labels": train_data.refund_labels,
            "delay_pay_labels_afterPay" : train_data.delay_pay_labels_afterPay,
            "delay_pay_label_afterRefund" : train_data.delay_pay_label_afterRefund,
            "inw_pay_labels_afterPay" : train_data.inw_pay_labels_afterPay,
            "delay_refund_label_afterRefund" : train_data.delay_refund_label_afterRefund,
            "inw_pay_labels_afterRefund" : train_data.inw_pay_labels_afterRefund,
            "stream_pay_labels" : train_data.stream_pay_labels,
            "stream_net_pay_labels": train_data.stream_net_pay_labels,
            "stream_pay_mask": train_data.stream_pay_mask
        },
        "test": {
            "features": test_data.features,
            "click_ts": test_data.click_ts,
            "pay_ts": test_data.pay_ts,
            "sample_ts": test_data.sample_ts,
            "refund_ts": test_data.refund_ts,
            "pay_labels": test_data.pay_labels,
            "net_pay_labels": test_data.net_pay_labels,
            "refund_labels": test_data.refund_labels,
            "delay_pay_labels_afterPay" : test_data.delay_pay_labels_afterPay,
            "delay_pay_label_afterRefund" : test_data.delay_pay_label_afterRefund,
            "inw_pay_labels_afterPay" : test_data.inw_pay_labels_afterPay,
            "delay_refund_label_afterRefund" : test_data.delay_refund_label_afterRefund,
            "inw_pay_labels_afterRefund" : test_data.inw_pay_labels_afterRefund,
            "stream_pay_labels" : test_data.stream_pay_labels,
            "stream_net_pay_labels": test_data.stream_net_pay_labels,
            "stream_pay_mask": test_data.stream_pay_mask
        }
    }


def get_ali_dataset_stream(args):
    """
    从指定路径加载或构建 Criteo 数据集，并返回训练和测试数据。

    Args:
        args (argparse.Namespace): 包含数据集名称、模式、数据缓存路径、等待窗口、属性窗口、训练分割起始和结束天数、测试分割起始和结束天数等参数。

    Returns:
        dict: 包含训练和测试数据的字典，其中训练数据以流的形式给出，测试数据以字典形式给出。

    """
    np.random.seed(args.seed)
    dataset_name = args.dataset_name
    mode = args.mode
    print("Loading data from {}".format(args.data_cache_path))
    cache_path = os.path.join(args.data_cache_path, f"{dataset_name}_{mode}.pkl")
    if os.path.isfile(cache_path):
        print("cache_path {} exists.".format(cache_path))
        print("loading cached data...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        main_train_stream = data["main_train_stream"]
        feature_train_stream = data["feature_train_stream"]
        unbias_train_stream = data["unbias_train_stream"]
        test_stream = data["test_stream"]
    
    else:
        print("building datasets")
        df,click_ts,pay_ts,refund_ts = get_data_df(args)
        pay_wait_window = args.pay_wait_window * SECONDS_A_DAY
        refund_wait_window = args.refund_wait_window * SECONDS_A_DAY
        pay_attr_window = args.pay_attr_window * SECONDS_A_DAY
        refund_attr_window = args.refund_attr_window * SECONDS_A_DAY
        data_src = DataDF(features = df, click_ts=click_ts, pay_ts=pay_ts, refund_ts=refund_ts,pay_attr_window=pay_attr_window,refund_attr_window=refund_attr_window)

        main_train_stream = []
        feature_train_stream = []
        unbias_train_stream = []
        test_stream = []
        print("splitting into train and test sets")

        if mode =="dfsn_train_stream":
            main_tower_train_data= data_src.sub_days(0, args.train_split_days_end).add_dfsn_main_duplicate_samples(pay_wait_window=pay_wait_window)
            main_tower_train_data= main_tower_train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            feature_tower_train_data= data_src.sub_days(0, args.train_split_days_end).add_dfsn_feature_duplicate_samples(pay_wait_window=pay_wait_window/2)
            feature_tower_train_data= feature_tower_train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            unbias_tower_train_data= data_src.sub_days(0, args.train_split_days_end).add_dfsn_unbias_duplicate_samples()
            unbias_tower_train_data= unbias_tower_train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end) 
     
        else:
            raise ValueError(f"Unknown mode {mode}")
        for i in np.arange(args.train_split_days_start,args.train_split_days_end,args.stream_wait_window):
            main_train_day = main_tower_train_data.sub_days(i,i+args.stream_wait_window)
            main_train_stream.append({"features": main_train_day.features,
                                    "click_ts": main_train_day.click_ts,
                                    "pay_ts": main_train_day.pay_ts,
                                    "sample_ts": main_train_day.sample_ts,
                                    "refund_ts": main_train_day.refund_ts,
                                    "pay_labels": main_train_day.pay_labels,
                                    "net_pay_labels": main_train_day.net_pay_labels,
                                    "refund_labels": main_train_day.refund_labels,
                                    "delay_pay_labels_afterPay" : main_train_day.delay_pay_labels_afterPay,
                                    "delay_pay_label_afterRefund" : main_train_day.delay_pay_label_afterRefund,
                                    "inw_pay_labels_afterPay" : main_train_day.inw_pay_labels_afterPay,
                                    "delay_refund_label_afterRefund" : main_train_day.delay_refund_label_afterRefund,
                                    "inw_pay_labels_afterRefund" : main_train_day.inw_pay_labels_afterRefund,
                                    "stream_pay_labels": main_train_day.stream_pay_labels,
                                    "stream_net_pay_labels": main_train_day.stream_net_pay_labels,
                                    "stream_pay_mask": main_train_day.stream_pay_mask})
            
            feature_train_day = feature_tower_train_data.sub_days(i,i+args.stream_wait_window)
            feature_train_stream.append({"features": feature_train_day.features,
                                    "click_ts": feature_train_day.click_ts,
                                    "pay_ts": feature_train_day.pay_ts,
                                    "sample_ts": feature_train_day.sample_ts,
                                    "refund_ts": feature_train_day.refund_ts,
                                    "pay_labels": feature_train_day.pay_labels,
                                    "net_pay_labels": feature_train_day.net_pay_labels,
                                    "refund_labels": feature_train_day.refund_labels,
                                    "delay_pay_labels_afterPay" : feature_train_day.delay_pay_labels_afterPay,
                                    "delay_pay_label_afterRefund" : feature_train_day.delay_pay_label_afterRefund,
                                    "inw_pay_labels_afterPay" : feature_train_day.inw_pay_labels_afterPay,
                                    "delay_refund_label_afterRefund" : feature_train_day.delay_refund_label_afterRefund,
                                    "inw_pay_labels_afterRefund" : feature_train_day.inw_pay_labels_afterRefund,
                                    "stream_pay_labels": feature_train_day.stream_pay_labels,
                                    "stream_net_pay_labels": feature_train_day.stream_net_pay_labels,
                                    "stream_pay_mask": feature_train_day.stream_pay_mask})

            unbias_train_day = unbias_tower_train_data.sub_days(i,i+args.stream_wait_window)
            unbias_train_stream.append({"features": unbias_train_day.features,
                                    "click_ts": unbias_train_day.click_ts,
                                    "pay_ts": unbias_train_day.pay_ts,
                                    "sample_ts": unbias_train_day.sample_ts,
                                    "refund_ts": unbias_train_day.refund_ts,
                                    "pay_labels": unbias_train_day.pay_labels,
                                    "net_pay_labels": unbias_train_day.net_pay_labels,
                                    "refund_labels": unbias_train_day.refund_labels,
                                    "delay_pay_labels_afterPay" : unbias_train_day.delay_pay_labels_afterPay,
                                    "delay_pay_label_afterRefund" : unbias_train_day.delay_pay_label_afterRefund,
                                    "inw_pay_labels_afterPay" : unbias_train_day.inw_pay_labels_afterPay,
                                    "delay_refund_label_afterRefund" : unbias_train_day.delay_refund_label_afterRefund,
                                    "inw_pay_labels_afterRefund" : unbias_train_day.inw_pay_labels_afterRefund,
                                    "stream_pay_labels": unbias_train_day.stream_pay_labels,
                                    "stream_net_pay_labels": unbias_train_day.stream_net_pay_labels,
                                    "stream_pay_mask": unbias_train_day.stream_pay_mask})


        for i in np.arange(args.test_split_days_start,args.test_split_days_end,args.stream_wait_window):
            test_day = test_data.sub_days(i,i+args.stream_wait_window)
            test_stream.append({"features": test_day.features,
                                "click_ts": test_day.click_ts,
                                "pay_ts": test_day.pay_ts,
                                "sample_ts": test_day.sample_ts,
                                "refund_ts": test_day.refund_ts,
                                "pay_labels": test_day.pay_labels,
                                "net_pay_labels": test_day.net_pay_labels,
                                "refund_labels": test_day.refund_labels,
                                "delay_pay_labels_afterPay" : test_day.delay_pay_labels_afterPay,
                                "delay_pay_label_afterRefund" : test_day.delay_pay_label_afterRefund,
                                "inw_pay_labels_afterPay" : test_day.inw_pay_labels_afterPay,
                                "delay_refund_label_afterRefund": test_day.delay_refund_label_afterRefund,
                                "inw_pay_labels_afterRefund": test_day.inw_pay_labels_afterRefund,
                                "stream_pay_labels": test_day.stream_pay_labels,
                                "stream_net_pay_labels": test_day.stream_net_pay_labels,
                                "stream_pay_mask": test_day.stream_pay_mask})
            
        print("writing data to cache file")
        if cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"main_train_stream": main_train_stream,"feature_train_stream": feature_train_stream,"unbias_train_stream": unbias_train_stream,"test_stream": test_stream}, f)

    print("====== Train SET ======")
    for day in range(len(main_train_stream)):
        print("Day",day)
        print(f"Total samples                : {len(main_train_stream[day]['pay_labels']):,}")
        print(f"Positive pay labels         : {sum(main_train_stream[day]['pay_labels']):,}")
        print(f"Positive net pay labels      : {sum(main_train_stream[day]['net_pay_labels']):,}")
        print(f"Positive refund labels       : {sum(main_train_stream[day]['refund_labels']):,}")
    print("====== Test SET ======")
    for day in range(len(test_stream)):
        print("Day",day)
        print(f"Total samples                : {len(test_stream[day]['pay_labels']):,}")
        print(f"Positive pay labels         : {sum(test_stream[day]['pay_labels']):,}")
        print(f"Positive net pay labels      : {sum(test_stream[day]['net_pay_labels']):,}")
        print(f"Positive refund labels       : {sum(test_stream[day]['refund_labels']):,}")

    return {"main_train_stream": main_train_stream,"feature_train_stream": feature_train_stream,"unbias_train_stream": unbias_train_stream,"test_stream": test_stream}
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/criteo/processed_data.txt', help='Data path')
    parser.add_argument('--data_cache_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/', help='Data path')
    parser.add_argument('--dataset_name', type=str, default='Criteo', help='Dataset name')
    parser.add_argument('--pay_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--refund_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--pay_wait_window', type=int, default=1, help='pay wait window size (days)')
    parser.add_argument('--refund_wait_window', type=int, default=1, help='refund wait window size (days)')
    parser.add_argument('--stream_wait_window', type=int, default=1, help='stream wait window size (days)')
    parser.add_argument('--train_split_days_start', type=int, default=0, help='start day of train (days)')
    parser.add_argument('--train_split_days_end', type=int, default=15, help='end day of train (days)')
    parser.add_argument('--test_split_days_start', type=int, default=15, help='start day of test (days)')
    parser.add_argument('--test_split_days_end', type=int, default=30, help='end day of test (days)')
    parser.add_argument('--mode', type=str, default="dfsn_pretrain", help='[dfsn_pretrain, dfsn_train_stream]')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    data_src = get_ali_dataset_stream(args)
    train_stream = data_src['train']
    for i in range(len(train_stream)):
        print(i)
        print('pos samples',sum(train_stream[i]['pay_labels']))
        print('total samples',len(train_stream[i]['pay_labels']))
        print('delay pos samples',sum(train_stream[i]['delay_pay_labels_afterPay']))

