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

    def __init__(self, features, click_ts, pay_ts, refund_ts, sample_ts=None, pay_labels=None,net_pay_labels =None,refund_labels=None,stream_pay_labels=None,stream_net_pay_labels=None,stream_pay_mask =None,delay_pay_labels_afterPay=None, delay_pay_label_afterRefund = None,delay_refund_label_afterRefund=None,inw_pay_labels_afterPay= None, inw_pay_labels_afterRefund=None, pay_attr_window = None, refund_attr_window = None):
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
            stream_pay_labels (numpy.ndarray, optional): 流转化标签数组，默认为 None。如果为 None，则使用转化标签作为流转化标签。
            stream_net_pay_labels (numpy.ndarray, optional): 流净转化标签数组，默认为 None。如果为 None，则使用净转化标签作为流净转化标签。
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
    
    def sub_days_inwspay_tn(self, start_day, end_day, pay_wait_window):
        """
        根据给定的 wait_window,attr_window提取指定日期范围内的数据。
        V2版本我们开始考虑实现窗口内转化标签(不考虑退款)
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
            inwpay_mask_afterPay = np.logical_and(self.pay_labels > 0,diff < pay_wait_window)
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, np.logical_and(diff > pay_wait_window, diff < self.pay_attr_window))
        else:
            inwpay_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts < pay_wait_window)
            delay_mask_afterPay = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > pay_wait_window)
        
        inw_pay_labels_afterPay = copy.deepcopy(self.pay_labels)
        inw_pay_labels_afterPay[~inwpay_mask_afterPay] = 0

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
                      inw_pay_labels_afterPay = inw_pay_labels_afterPay[mask],
                      delay_pay_labels_afterPay = delay_pay_labels_afterPay[mask],
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

    def add_defer_duplicate_samples(self, pay_wait_window):
        """
        在给定归因窗口和观测窗口内，生成带有延迟标签的样本，并返回DataDF对象。注意此时我们不考虑退款

        Args:
            pay_wait_window (int): 观测窗口的大小，表示从点击到支付的最大等待时间

        Returns:
            DataDF: 包含新生成的带有延迟标签的样本的DataDF对象。

        """
        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        pay_label_mask = np.logical_and(self.pay_ts > 0, self.pay_labels)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
    
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[pay_label_mask],
        df2[~pay_label_mask]
        ])

        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[pay_label_mask],
            self.click_ts[~pay_label_mask] + self.pay_attr_window
        ])
        click_ts = np.concatenate([self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask], \
            self.click_ts[pay_label_mask], self.click_ts[~pay_label_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask], \
            self.pay_ts[pay_label_mask], self.pay_ts[~pay_label_mask]], axis=0)
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], \
                        self.refund_ts[pay_label_mask], self.refund_ts[~pay_label_mask]], axis=0)
    
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),
        np.zeros((np.sum(~inw_pay_mask),)),
        pay_labels[pay_label_mask],
        pay_labels[~pay_label_mask]
        ], axis=0)

        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[pay_label_mask],                
        pay_labels[~pay_label_mask]                
        ], axis=0)

        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[pay_label_mask],              
        net_pay_labels[~pay_label_mask] 
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[pay_label_mask],             
        refund_labels[~pay_label_mask]  
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

    def add_defer_duplicate_samples_v2(self, pay_wait_window):
        """
        在给定归因窗口和观测窗口内，生成带有延迟标签的样本，并返回DataDF对象。注意此时我们不考虑退款

        Args:
            pay_wait_window (int): 观测窗口的大小，表示从点击到支付的最大等待时间

        Returns:
            DataDF: 包含新生成的带有延迟标签的样本的DataDF对象。

        """
        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        pay_label_mask = np.logical_and(self.pay_ts > 0, self.pay_labels)
        delay_pay_mask = np.logical_and((self.pay_ts - self.click_ts) > pay_wait_window,self.pay_labels)
        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
    
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        df2[~pay_label_mask]
        ])

        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask],
            self.click_ts[~pay_label_mask] + self.pay_attr_window
        ])
        click_ts = np.concatenate([self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask], \
            self.click_ts[delay_pay_mask], self.click_ts[~pay_label_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask], \
            self.pay_ts[delay_pay_mask], self.pay_ts[~pay_label_mask]], axis=0)
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], \
                        self.refund_ts[delay_pay_mask], self.refund_ts[~pay_label_mask]], axis=0)
    
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),
        np.zeros((np.sum(~inw_pay_mask),)),
        pay_labels[delay_pay_mask],
        pay_labels[~pay_label_mask]
        ], axis=0)

        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask],                
        pay_labels[~pay_label_mask]                
        ], axis=0)

        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask],              
        net_pay_labels[~pay_label_mask] 
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask],             
        refund_labels[~pay_label_mask]  
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

    def add_vanilla_pay_inw_samples(self,pay_wait_window):
        """
        只考虑窗口内的转化样本（即立即支付样本），不考虑延迟支付样本。

        Args:
            pay_wait_window (float): 支付等待窗口时间，用于判断支付是否延迟。

        Returns:
            DataDF: 包含伪负样本的新数据集。

            """
        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        df1 = self.features.copy(deep=True)

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
        np.zeros((np.sum(~inw_pay_mask),)),
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

    def add_vanilla_netpay_inw_samples(self,pay_wait_window, refund_wait_window):
        """
        只考虑窗口内的转化样本（即立即支付样本），净转化样本，不做样本回补

        Args:
            pay_wait_window (float): 支付等待窗口时间，用于判断支付是否延迟。

        Returns:
            DataDF: 包含伪负样本的新数据集。

            """
        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_refund_mask = np.logical_and(np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) > refund_wait_window),self.refund_labels)

        df1 = self.features.copy(deep=True)

        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        ])
        

        sample_ts = np.concatenate([
            self.pay_ts[inw_pay_mask] + refund_wait_window,
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
        np.zeros((np.sum(~inw_pay_mask),)),
        ], axis=0)

        net_pay_labels_inw_pay =  copy.deepcopy(self.net_pay_labels)
        net_pay_labels_inw_pay[delay_refund_mask] = 1
        stream_net_pay_labels = np.concatenate([
        net_pay_labels_inw_pay[inw_pay_mask],
        np.zeros((np.sum(~inw_pay_mask),)),
        ], axis=0)

        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)
    
    def add_esdfm_fake_neg(self,pay_wait_window):
        """
        为数据集添加延迟支付的伪负样本。

        Args:
            pay_wait_window (float): 支付等待窗口时间，用于判断支付是否延迟。

        Returns:
            DataDF: 包含伪负样本的新数据集。

            """
        inw_pay_mask = np.logical_and((self.pay_ts - self.click_ts) <= pay_wait_window,self.pay_labels)
        delay_pay_mask = np.logical_and((self.pay_ts - self.click_ts) > pay_wait_window,self.pay_labels)
        new_features = pd.concat([
        self.features[inw_pay_mask],
        self.features[~inw_pay_mask],
        self.features.iloc[delay_pay_mask].copy(deep=True)
        ])
        sample_ts = np.concatenate(
            [self.click_ts[inw_pay_mask] + pay_wait_window,
             self.click_ts[~inw_pay_mask] + pay_wait_window,
             self.pay_ts[delay_pay_mask]], axis=0
        )
        click_ts = np.concatenate(
            [self.click_ts[inw_pay_mask], 
             self.click_ts[~inw_pay_mask],
             self.click_ts[delay_pay_mask]], axis=0
        )
        pay_ts = np.concatenate(
            [self.pay_ts[inw_pay_mask], 
             self.pay_ts[~inw_pay_mask],
             self.pay_ts[delay_pay_mask]], axis=0
        )
        refund_ts = np.concatenate(
            [self.refund_ts[inw_pay_mask], 
             self.refund_ts[~inw_pay_mask],
             self.refund_ts[delay_pay_mask]], axis=0
        )

        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)


        stream_pay_labels = np.concatenate([
            np.ones((np.sum(inw_pay_mask),)),
            np.zeros((np.sum(~inw_pay_mask),)),
            np.ones((np.sum(delay_pay_mask),))
        ], axis=0)


        pay_labels = np.concatenate([pay_labels[inw_pay_mask], 
                                     pay_labels[~inw_pay_mask],
                                     pay_labels[delay_pay_mask]],axis=0)
        
        net_pay_labels = np.concatenate([net_pay_labels[inw_pay_mask], 
                                         net_pay_labels[~inw_pay_mask],
                                         net_pay_labels[delay_pay_mask]],axis=0)
        
        refund_labels = np.concatenate([refund_labels[inw_pay_mask],
                                        refund_labels[~inw_pay_mask],
                                        refund_labels[delay_pay_mask]],axis=0)


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

    def add_defuse_fake_neg(self,pay_wait_window):
        """
        为数据集添加延迟支付的伪负样本。

        Args:
            pay_wait_window (float): 支付等待窗口时间，用于判断支付是否延迟。

        Returns:
            DataDF: 包含伪负样本的新数据集。

            """
        inw_pay_mask = np.logical_and((self.pay_ts - self.click_ts) <= pay_wait_window,self.pay_labels)
        delay_pay_mask = np.logical_and((self.pay_ts - self.click_ts) > pay_wait_window,self.pay_labels)
        new_features = pd.concat([
        self.features[inw_pay_mask],
        self.features[~inw_pay_mask],
        self.features.iloc[delay_pay_mask].copy(deep=True)
        ])
        sample_ts = np.concatenate(
            [self.click_ts[inw_pay_mask] + pay_wait_window,
             self.click_ts[~inw_pay_mask] + pay_wait_window,
             self.pay_ts[delay_pay_mask]], axis=0
        )
        click_ts = np.concatenate(
            [self.click_ts[inw_pay_mask], 
             self.click_ts[~inw_pay_mask],
             self.click_ts[delay_pay_mask]], axis=0
        )
        pay_ts = np.concatenate(
            [self.pay_ts[inw_pay_mask], 
             self.pay_ts[~inw_pay_mask],
             self.pay_ts[delay_pay_mask]], axis=0
        )
        refund_ts = np.concatenate(
            [self.refund_ts[inw_pay_mask], 
             self.refund_ts[~inw_pay_mask],
             self.refund_ts[delay_pay_mask]], axis=0
        )

        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)


        stream_pay_labels = np.concatenate([
            np.ones((np.sum(inw_pay_mask),)),
            np.zeros((np.sum(~inw_pay_mask),)),
            np.ones((np.sum(delay_pay_mask),))
        ], axis=0)


        pay_labels = np.concatenate([pay_labels[inw_pay_mask], 
                                     pay_labels[~inw_pay_mask],
                                     pay_labels[delay_pay_mask]],axis=0)
        
        net_pay_labels = np.concatenate([net_pay_labels[inw_pay_mask], 
                                         net_pay_labels[~inw_pay_mask],
                                         net_pay_labels[delay_pay_mask]],axis=0)
        
        refund_labels = np.concatenate([refund_labels[inw_pay_mask],
                                        refund_labels[~inw_pay_mask],
                                        refund_labels[delay_pay_mask]],axis=0)


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

    def add_fnw_fake_neg(self):
        """
        添加假负样本以处理延迟转化。
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

    def add_fnc_fake_neg(self):
        """
        添加假负样本以处理延迟转化。
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

    def add_oracle_samples(self):
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

    def add_bidefuse_inw_outw_delay_postitive(self,pay_wait_window):

        inw_pay_mask = np.logical_and((self.pay_ts - self.click_ts) <= pay_wait_window,self.pay_labels)
        delay_pay_mask = np.logical_and((self.pay_ts - self.click_ts) > pay_wait_window,self.pay_labels)
    
        new_features = pd.concat([
        self.features[inw_pay_mask],
        self.features[~inw_pay_mask],
        self.features.iloc[delay_pay_mask].copy(deep=True)
        ])

        sample_ts = np.concatenate(
            [self.click_ts[inw_pay_mask]+ pay_wait_window,
             self.click_ts[~inw_pay_mask] + pay_wait_window,
             self.pay_ts[delay_pay_mask]], axis=0)
        
        click_ts = np.concatenate(
            [self.click_ts[inw_pay_mask],
             self.click_ts[~inw_pay_mask],
             self.click_ts[delay_pay_mask]], axis=0)
        pay_ts = np.concatenate(
            [self.pay_ts[inw_pay_mask],
             self.pay_ts[~inw_pay_mask],
             self.pay_ts[delay_pay_mask]], axis=0)
        refund_ts = np.concatenate(
            [self.refund_ts[inw_pay_mask], 
             self.refund_ts[~inw_pay_mask],
             self.refund_ts[delay_pay_mask]], axis=0)
        
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = copy.deepcopy(self.pay_labels)
        stream_pay_labels[delay_pay_mask] = 0
        stream_pay_labels = np.concatenate([
            np.ones((np.sum(inw_pay_mask),)),
            np.zeros((np.sum(~inw_pay_mask),)),
            np.ones((np.sum(delay_pay_mask),))
        ], axis=0)


        delay_pay_labels = np.concatenate([np.zeros(pay_labels[inw_pay_mask].shape[0]), np.zeros((np.sum(~inw_pay_mask),)), np.ones((np.sum(delay_pay_mask),))], axis=0)
        inw_pay_labels = np.concatenate([np.ones(pay_labels[inw_pay_mask].shape[0]), np.zeros((np.sum(~inw_pay_mask),)), np.zeros((np.sum(delay_pay_mask),))], axis=0)

        pay_labels = np.concatenate([pay_labels[inw_pay_mask],pay_labels[~inw_pay_mask], pay_labels[delay_pay_mask]],axis=0)
        net_pay_labels = np.concatenate([net_pay_labels[inw_pay_mask],net_pay_labels[~inw_pay_mask], net_pay_labels[delay_pay_mask]],axis=0)
        refund_labels = np.concatenate([refund_labels[inw_pay_mask],refund_labels[~inw_pay_mask], refund_labels[delay_pay_mask]],axis=0)

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
                      delay_pay_labels_afterPay=delay_pay_labels[idx],
                      inw_pay_labels_afterPay=inw_pay_labels[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_redefer_duplicate_samples(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        v0版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_labels, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        second_pay_mask = np.logical_or(inw_pay_mask, delay_pay_mask)

        refund_mask = np.logical_and(self.refund_ts > 0,self.refund_labels)
        third_pay_mask = np.logical_and(self.pay_ts>0,self.net_pay_labels)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[second_pay_mask],
        df2[~second_pay_mask],
        df3[refund_mask],
        ])
                

        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[second_pay_mask],
            self.click_ts[~second_pay_mask] + self.pay_attr_window,
            self.refund_ts[refund_mask]  ,
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], 
            self.click_ts[~inw_pay_mask],
            self.click_ts[second_pay_mask],
            self.click_ts[~second_pay_mask],
            self.click_ts[refund_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], 
            self.pay_ts[~inw_pay_mask],
            self.pay_ts[second_pay_mask], 
            self.pay_ts[~second_pay_mask],
            self.pay_ts[refund_mask]], axis=0)
        
        refund_ts = np.concatenate([
            self.refund_ts[inw_pay_mask], 
            self.refund_ts[~inw_pay_mask], 
            self.refund_ts[second_pay_mask], 
            self.refund_ts[~second_pay_mask],
            self.refund_ts[refund_mask]], axis=0)



        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(second_pay_mask),)),     
        np.zeros((np.sum(~second_pay_mask))),   
        np.ones((np.sum(refund_mask),)),    
        ], axis=0)

        stream_net_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.zeros((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(second_pay_mask),)),
        np.zeros((np.sum(~second_pay_mask))),
        net_pay_labels[refund_mask],
        ], axis=0)


        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(second_pay_mask),)),
        np.ones((np.sum(~second_pay_mask))),
        np.zeros((np.sum(refund_mask))),
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[second_pay_mask],               
        pay_labels[~second_pay_mask],                
        pay_labels[refund_mask],                               
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[second_pay_mask],     
        net_pay_labels[~second_pay_mask],     
        net_pay_labels[refund_mask],     
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[second_pay_mask],          
        refund_labels[~second_pay_mask],          
        refund_labels[refund_mask],          
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_redefer_duplicate_samples_v1(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        v0版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_labels, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        second_pay_mask = np.logical_or(inw_pay_mask, delay_pay_mask)

        refund_mask = np.logical_and(self.refund_ts > 0,self.refund_labels)
        third_pay_mask = np.logical_and(self.pay_ts>0,self.net_pay_labels)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        df2[~second_pay_mask],
        df3[refund_mask],
        ])
        

    
        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask],
            self.click_ts[~second_pay_mask] + self.pay_attr_window,
            self.refund_ts[refund_mask]  ,
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], 
            self.click_ts[~inw_pay_mask],
            self.click_ts[delay_pay_mask],
            self.click_ts[~second_pay_mask],
            self.click_ts[refund_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], 
            self.pay_ts[~inw_pay_mask],
            self.pay_ts[delay_pay_mask], 
            self.pay_ts[~second_pay_mask],
            self.pay_ts[refund_mask]], axis=0)
        
        refund_ts = np.concatenate([
            self.refund_ts[inw_pay_mask], 
            self.refund_ts[~inw_pay_mask], 
            self.refund_ts[delay_pay_mask], 
            self.refund_ts[~second_pay_mask],
            self.refund_ts[refund_mask]], axis=0)



        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(delay_pay_mask),)),     
        np.zeros((np.sum(~second_pay_mask))),   
        np.ones((np.sum(refund_mask),)),       
        ], axis=0)

        stream_net_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),   
        np.zeros((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),
        np.zeros((np.sum(~second_pay_mask))),
        net_pay_labels[refund_mask],
        ], axis=0)


        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),
        np.ones((np.sum(~second_pay_mask))),
        np.zeros((np.sum(refund_mask))),
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask],               
        pay_labels[~second_pay_mask],                
        pay_labels[refund_mask],                               
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask],     
        net_pay_labels[~second_pay_mask],     
        net_pay_labels[refund_mask],     
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask],          
        refund_labels[~second_pay_mask],          
        refund_labels[refund_mask],          
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_redefer_v0_duplicate_samples_wo_refund_backfill(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        v0版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        second_pay_mask = np.logical_or(inw_net_pay_mask, delay_pay_mask)

        refund_mask = np.logical_and(np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) > refund_wait_window),self.refund_labels)
        third_pay_mask = np.logical_and(self.pay_ts>0,self.net_pay_labels)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[second_pay_mask],
        df2[~second_pay_mask],
        ])
        

        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[second_pay_mask],
            self.click_ts[~second_pay_mask] + self.pay_attr_window,
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], 
            self.click_ts[~inw_pay_mask],
            self.click_ts[second_pay_mask],
            self.click_ts[~second_pay_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], 
            self.pay_ts[~inw_pay_mask],
            self.pay_ts[second_pay_mask], 
            self.pay_ts[~second_pay_mask]], axis=0)
        
        refund_ts = np.concatenate([
            self.refund_ts[inw_pay_mask], 
            self.refund_ts[~inw_pay_mask], 
            self.refund_ts[second_pay_mask], 
            self.refund_ts[~second_pay_mask]], axis=0)



        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(second_pay_mask),)),     
        np.zeros((np.sum(~second_pay_mask))),     
        ], axis=0)




        stream_net_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.zeros((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(second_pay_mask),)),
        np.zeros((np.sum(~second_pay_mask))),
        ], axis=0)


        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(second_pay_mask),)),
        np.ones((np.sum(~second_pay_mask))),
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[second_pay_mask],               
        pay_labels[~second_pay_mask],                                         
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[second_pay_mask],     
        net_pay_labels[~second_pay_mask],     
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[second_pay_mask],          
        refund_labels[~second_pay_mask],                 
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_redefer_v1_duplicate_samples(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补

        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)

        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        second_pay_mask = np.logical_or(inw_net_pay_mask, delay_pay_mask)

        delay_refund_mask = np.logical_and(np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) > refund_wait_window),self.refund_labels)
        third_pay_mask = np.logical_and(self.pay_ts>0,self.net_pay_labels)


        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_net_pay_mask],
        df1[~inw_net_pay_mask],
        df2[second_pay_mask],
        df2[~second_pay_mask],
        df3[third_pay_mask],
        df3[~third_pay_mask]
        ])

        sample_ts_not_inw_net_pay_mask = np.where(
            self.refund_ts[~inw_net_pay_mask] > 0,
            self.refund_ts[~inw_net_pay_mask] ,
            self.click_ts[~inw_net_pay_mask] + pay_wait_window 
        )
        
        sample_ts_net_pay_neg = np.where(
            self.refund_ts[~third_pay_mask] > 0,
            self.refund_ts[~third_pay_mask] ,
            self.click_ts[~third_pay_mask] + self.pay_attr_window + self.refund_attr_window
        )


        sample_ts = np.concatenate([
            self.pay_ts[inw_net_pay_mask] + refund_wait_window,
            sample_ts_not_inw_net_pay_mask,
            self.pay_ts[second_pay_mask],
            self.click_ts[~second_pay_mask] + self.pay_attr_window,
            self.pay_ts[third_pay_mask] + self.refund_attr_window ,
            sample_ts_net_pay_neg,
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_net_pay_mask], self.click_ts[~inw_net_pay_mask],
            self.click_ts[second_pay_mask], self.click_ts[~second_pay_mask],
            self.click_ts[third_pay_mask],self.click_ts[~third_pay_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_net_pay_mask], self.pay_ts[~inw_net_pay_mask],
            self.pay_ts[second_pay_mask], self.pay_ts[~second_pay_mask],
            self.pay_ts[third_pay_mask], self.pay_ts[~third_pay_mask]], axis=0)
        
        refund_ts = np.concatenate([self.refund_ts[inw_net_pay_mask], self.refund_ts[~inw_net_pay_mask], 
            self.refund_ts[second_pay_mask], self.refund_ts[~second_pay_mask],
            self.refund_ts[third_pay_mask], self.refund_ts[~third_pay_mask]], axis=0)



        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_net_pay_mask),)),      
        np.zeros((np.sum(~inw_net_pay_mask),)),     
        np.ones((np.sum(second_pay_mask),)),     
        np.zeros((np.sum(~second_pay_mask))),     
        net_pay_labels[third_pay_mask],
        net_pay_labels[~third_pay_mask]
        ], axis=0)

        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_net_pay_mask),)),
        np.ones((np.sum(~inw_net_pay_mask),)),
        np.ones((np.sum(second_pay_mask),)),
        np.ones((np.sum(~second_pay_mask))),
        np.zeros((np.sum(third_pay_mask))),
        np.zeros((np.sum(~third_pay_mask)))
        ], axis=0)
        pay_labels = np.concatenate([
        pay_labels[inw_net_pay_mask],               
        pay_labels[~inw_net_pay_mask],                
        pay_labels[second_pay_mask],               
        pay_labels[~second_pay_mask],                
        pay_labels[third_pay_mask],                 
        pay_labels[~third_pay_mask]                  
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_net_pay_mask],          
        net_pay_labels[~inw_net_pay_mask],     
        net_pay_labels[second_pay_mask],     
        net_pay_labels[~second_pay_mask],     
        net_pay_labels[third_pay_mask],     
        net_pay_labels[~third_pay_mask]
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[second_pay_mask],          
        refund_labels[~second_pay_mask],          
        refund_labels[third_pay_mask],          
        refund_labels[~third_pay_mask]
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
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)
    
    def add_redefer_v2_duplicate_samples(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        从v2版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        second_pay_mask = np.logical_or(inw_net_pay_mask, delay_pay_mask)

        delay_refund_mask = np.logical_and(np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) > refund_wait_window),self.refund_labels)
        third_pay_mask = np.logical_and(self.pay_ts>0,self.net_pay_labels)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[second_pay_mask],
        df2[~second_pay_mask],
        df3[third_pay_mask],
        df3[~third_pay_mask]
        ])
        
        inw_refund_ts = copy.deepcopy(self.refund_ts)
        inw_refund_ts[~inw_refund_mask] = -1
        sample_ts_inw_pay_mask = np.where(
            inw_refund_ts[inw_pay_mask] > 0,
            self.refund_ts[inw_pay_mask] ,
            self.pay_ts[inw_pay_mask] + refund_wait_window 
        )
        
        sample_ts_net_pay_neg = np.where(
            self.refund_ts[~third_pay_mask] > 0,
            self.refund_ts[~third_pay_mask] ,
            self.click_ts[~third_pay_mask] + self.pay_attr_window + self.refund_attr_window
        )
        sample_ts = np.concatenate([
            sample_ts_inw_pay_mask,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[second_pay_mask],
            self.click_ts[~second_pay_mask] + self.pay_attr_window,
            self.pay_ts[third_pay_mask] + self.refund_attr_window ,
            sample_ts_net_pay_neg,
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask],
            self.click_ts[second_pay_mask], self.click_ts[~second_pay_mask],
            self.click_ts[third_pay_mask],self.click_ts[~third_pay_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask],
            self.pay_ts[second_pay_mask], self.pay_ts[~second_pay_mask],
            self.pay_ts[third_pay_mask], self.pay_ts[~third_pay_mask]], axis=0)
        
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], 
            self.refund_ts[second_pay_mask], self.refund_ts[~second_pay_mask],
            self.refund_ts[third_pay_mask], self.refund_ts[~third_pay_mask]], axis=0)



        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(second_pay_mask),)),     
        np.zeros((np.sum(~second_pay_mask))),     
        net_pay_labels[third_pay_mask],
        net_pay_labels[~third_pay_mask]
        ], axis=0)

        net_pay_labels_inw_pay =  copy.deepcopy(self.net_pay_labels)
        net_pay_labels_inw_pay[delay_refund_mask] = 1
        stream_net_pay_labels = np.concatenate([
        net_pay_labels_inw_pay[inw_pay_mask],
        np.zeros((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(second_pay_mask),)),
        np.zeros((np.sum(~second_pay_mask))),
        net_pay_labels[third_pay_mask],
        net_pay_labels[~third_pay_mask]
        ], axis=0)


        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(second_pay_mask),)),
        np.ones((np.sum(~second_pay_mask))),
        np.zeros((np.sum(third_pay_mask))),
        np.zeros((np.sum(~third_pay_mask)))
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[second_pay_mask],               
        pay_labels[~second_pay_mask],                
        pay_labels[third_pay_mask],                 
        pay_labels[~third_pay_mask]                  
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[second_pay_mask],     
        net_pay_labels[~second_pay_mask],     
        net_pay_labels[third_pay_mask],     
        net_pay_labels[~third_pay_mask]
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[second_pay_mask],          
        refund_labels[~second_pay_mask],          
        refund_labels[third_pay_mask],          
        refund_labels[~third_pay_mask]
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_reesdfm_duplicate_samples(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        esdfm退款版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        refund_mask = np.logical_and(self.refund_ts > 0,self.refund_labels)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        df3[refund_mask],
        ])
        

        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask],
            self.refund_ts[refund_mask],
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask],
            self.click_ts[delay_pay_mask], self.click_ts[refund_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask],
            self.pay_ts[delay_pay_mask], self.pay_ts[refund_mask]], axis=0)
        
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], 
            self.refund_ts[delay_pay_mask], self.refund_ts[refund_mask]], axis=0)


        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(delay_pay_mask),)),     
        np.ones((np.sum(refund_mask))),     
        ], axis=0)


        stream_net_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),   
        np.zeros((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),     
        np.zeros((np.sum(refund_mask))),     
        ], axis=0)


        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),
        np.zeros((np.sum(refund_mask)))
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask], 
        pay_labels[refund_mask]
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask],     
        net_pay_labels[refund_mask],     
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask],          
        refund_labels[refund_mask],          
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_reesdfm_duplicate_samples_wo_refund_backfill(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        esdfm退款版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        delay_refund_mask = np.logical_and(np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) > refund_wait_window),self.refund_labels)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        ])
        

        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask],
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask],
            self.click_ts[delay_pay_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask],
            self.pay_ts[delay_pay_mask]], axis=0)
        
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], 
            self.refund_ts[delay_pay_mask]], axis=0)


        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(delay_pay_mask),)),     
        ], axis=0)

        net_pay_labels_inw_pay =  copy.deepcopy(self.net_pay_labels)
        net_pay_labels_inw_pay[delay_refund_mask] = 1
        stream_net_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),   
        np.zeros((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),        
        ], axis=0)


        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask], 
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask],     
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask],                 
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_redefuse_duplicate_samples(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        defuse退款版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        refund_mask = np.logical_and(self.refund_ts > 0,self.refund_labels)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        df3[refund_mask],
        ])
        

        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask],
            self.refund_ts[refund_mask],
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], self.click_ts[~inw_pay_mask],
            self.click_ts[delay_pay_mask], self.click_ts[refund_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], self.pay_ts[~inw_pay_mask],
            self.pay_ts[delay_pay_mask], self.pay_ts[refund_mask]], axis=0)
        
        refund_ts = np.concatenate([self.refund_ts[inw_pay_mask], self.refund_ts[~inw_pay_mask], 
            self.refund_ts[delay_pay_mask], self.refund_ts[refund_mask]], axis=0)


        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(delay_pay_mask),)),     
        np.ones((np.sum(refund_mask))),     
        ], axis=0)


        stream_net_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.zeros((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),     
        np.zeros((np.sum(refund_mask))),     
        ], axis=0)


        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),
        np.zeros((np.sum(refund_mask)))
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask], 
        pay_labels[refund_mask]
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask],     
        net_pay_labels[refund_mask],     
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask],          
        refund_labels[refund_mask],          
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_refnc_duplicate_samples(self):
        pos_mask = np.logical_and(self.pay_ts>0, self.pay_labels)
        refund_mask = np.logical_and(self.refund_ts>0, self.refund_labels)


        new_features = pd.concat([
        self.features.copy(deep=True), 
        self.features.iloc[pos_mask].copy(deep=True),
        self.features.iloc[refund_mask].copy(deep=True)
        ], axis=0)

        sample_ts = np.concatenate([
            self.click_ts,
            self.pay_ts[pos_mask],
            self.refund_ts[refund_mask]
        ])

        click_ts = np.concatenate([
            self.click_ts, self.click_ts[pos_mask], self.click_ts[refund_mask]
        ])
        pay_ts = np.concatenate([
            self.pay_ts, self.pay_ts[pos_mask], self.pay_ts[refund_mask]
        ])
        refund_ts = np.concatenate([
            self.refund_ts, self.refund_ts[pos_mask], self.refund_ts[refund_mask]
        ])
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)
        stream_pay_labels = copy.deepcopy(self.pay_labels)
        stream_pay_labels[pos_mask] = 0
        stream_pay_labels = np.concatenate([stream_pay_labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        stream_net_pay_labels = copy.deepcopy(self.net_pay_labels)
        stream_net_pay_labels[pos_mask] = 0
        stream_net_pay_labels = np.concatenate([stream_net_pay_labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        stream_pay_mask = np.concatenate([np.ones_like(self.pay_labels),np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        pay_labels = np.concatenate([pay_labels, pay_labels[pos_mask], pay_labels[refund_mask]],axis=0)
        net_pay_labels = np.concatenate([net_pay_labels, net_pay_labels[pos_mask], net_pay_labels[refund_mask]],axis=0)
        refund_labels = np.concatenate([refund_labels, refund_labels[pos_mask], refund_labels[refund_mask]],axis=0)


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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_base_duplicate_samples(self):
        pos_mask = np.logical_and(self.pay_ts>0, self.pay_labels)
        refund_mask = np.logical_and(self.refund_ts>0, self.refund_labels)
        new_features = pd.concat([
        self.features.copy(deep=True), 
        self.features.iloc[pos_mask].copy(deep=True),
        self.features.iloc[refund_mask].copy(deep=True)
        ], axis=0)

        sample_ts = np.concatenate([
            self.click_ts,
            self.pay_ts[pos_mask],
            self.refund_ts[refund_mask]
        ])

        click_ts = np.concatenate([
            self.click_ts, self.click_ts[pos_mask], self.click_ts[refund_mask]
        ])
        pay_ts = np.concatenate([
            self.pay_ts, self.pay_ts[pos_mask], self.pay_ts[refund_mask]
        ])
        refund_ts = np.concatenate([
            self.refund_ts, self.refund_ts[pos_mask], self.refund_ts[refund_mask]
        ])
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)
        stream_pay_labels = copy.deepcopy(self.pay_labels)
        stream_pay_labels[pos_mask] = 0
        stream_pay_labels = np.concatenate([stream_pay_labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        stream_net_pay_labels = copy.deepcopy(self.net_pay_labels)
        stream_net_pay_labels[pos_mask] = 0
        stream_net_pay_labels = np.concatenate([stream_net_pay_labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        stream_pay_mask = np.concatenate([np.ones_like(self.pay_labels),np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        pay_labels = np.concatenate([pay_labels, pay_labels[pos_mask], pay_labels[refund_mask]],axis=0)
        net_pay_labels = np.concatenate([net_pay_labels, net_pay_labels[pos_mask], net_pay_labels[refund_mask]],axis=0)
        refund_labels = np.concatenate([refund_labels, refund_labels[pos_mask], refund_labels[refund_mask]],axis=0)

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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_base_duplicate_samples_for_Preliminary_Experiment(self):
        pos_mask = np.logical_and(self.pay_ts>0, self.pay_labels)
        refund_mask = np.logical_and(self.refund_ts>0, self.refund_labels)
        new_features = pd.concat([
        self.features.copy(deep=True), 
        self.features.iloc[pos_mask].copy(deep=True),
        self.features.iloc[refund_mask].copy(deep=True)
        ], axis=0)

        sample_ts = np.concatenate([
            self.click_ts,
            self.pay_ts[pos_mask],
            self.refund_ts[refund_mask]
        ])

        click_ts = np.concatenate([
            self.click_ts, self.click_ts[pos_mask], self.click_ts[refund_mask]
        ])
        pay_ts = np.concatenate([
            self.pay_ts, self.pay_ts[pos_mask], self.pay_ts[refund_mask]
        ])
        refund_ts = np.concatenate([
            self.refund_ts, self.refund_ts[pos_mask], self.refund_ts[refund_mask]
        ])
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)
        stream_pay_labels = copy.deepcopy(self.pay_labels)
        stream_pay_labels[pos_mask] = 0
        stream_pay_labels = np.concatenate([
            stream_pay_labels, 
            np.ones((np.sum(pos_mask),)), 
            np.zeros((np.sum(refund_mask),))], axis=0)

        stream_net_pay_labels = copy.deepcopy(self.net_pay_labels)
        stream_net_pay_labels[pos_mask] = 0
        stream_net_pay_labels = np.concatenate([
            stream_net_pay_labels, 
            np.ones((np.sum(pos_mask),)), 
            np.zeros((np.sum(refund_mask),))], axis=0)

        stream_pay_mask = np.concatenate([
            np.ones_like(self.pay_labels),
            np.ones((np.sum(pos_mask),)), 
            np.zeros((np.sum(refund_mask),))], axis=0)
        
        delay_pay_labels_afterPay = np.concatenate([
            np.zeros_like(self.pay_labels), 
            np.ones((np.sum(pos_mask),)), 
            np.zeros((np.sum(refund_mask),))], axis=0)
        
        pay_labels = np.concatenate([pay_labels, pay_labels[pos_mask], pay_labels[refund_mask]],axis=0)
        net_pay_labels = np.concatenate([net_pay_labels, net_pay_labels[pos_mask], net_pay_labels[refund_mask]],axis=0)
        refund_labels = np.concatenate([refund_labels, refund_labels[pos_mask], refund_labels[refund_mask]],axis=0)

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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      delay_pay_labels_afterPay=delay_pay_labels_afterPay[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_reoracle_samples(self):

        pos_mask = np.logical_and(self.pay_ts>0, self.pay_labels)
        refund_mask = np.logical_and(self.refund_ts>0, self.refund_labels)


        new_features = pd.concat([
        self.features.copy(deep=True), 
        self.features.iloc[pos_mask].copy(deep=True),
        self.features.iloc[refund_mask].copy(deep=True)
        ], axis=0)

        sample_ts = np.concatenate([
            self.click_ts,
            self.pay_ts[pos_mask],
            self.refund_ts[refund_mask]
        ])

        click_ts = np.concatenate([
            self.click_ts, self.click_ts[pos_mask], self.click_ts[refund_mask]
        ])
        pay_ts = np.concatenate([
            self.pay_ts, self.pay_ts[pos_mask], self.pay_ts[refund_mask]
        ])
        refund_ts = np.concatenate([
            self.refund_ts, self.refund_ts[pos_mask], self.refund_ts[refund_mask]
        ])
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)
        stream_pay_labels = copy.deepcopy(self.pay_labels)
        stream_pay_labels[pos_mask] = 0
        stream_pay_labels = np.concatenate([stream_pay_labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        stream_net_pay_labels = copy.deepcopy(self.net_pay_labels)
        stream_net_pay_labels[pos_mask] = 0
        stream_net_pay_labels = np.concatenate([stream_net_pay_labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        stream_pay_mask = np.concatenate([np.ones_like(self.pay_labels),np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)

        pay_labels = np.concatenate([pay_labels, pay_labels[pos_mask], pay_labels[refund_mask]],axis=0)
        net_pay_labels = np.concatenate([net_pay_labels, net_pay_labels[pos_mask], net_pay_labels[refund_mask]],axis=0)
        refund_labels = np.concatenate([refund_labels, refund_labels[pos_mask], refund_labels[refund_mask]],axis=0)


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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_refnw_duplicate_samples(self):
        """
        假正样本(延迟退款)，假负样本(延迟转化) 回补

        Args:

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """


        pos_mask = np.logical_and(self.pay_ts>0, self.pay_labels)
        refund_mask = np.logical_and(self.refund_ts>0, self.refund_labels)


        new_features = pd.concat([
        self.features.copy(deep=True), 
        self.features.iloc[pos_mask].copy(deep=True),
        self.features.iloc[refund_mask].copy(deep=True)
        ], axis=0)

        sample_ts = np.concatenate([
            self.click_ts,
            self.pay_ts[pos_mask],
            self.refund_ts[refund_mask]
        ])
        click_ts = np.concatenate([
            self.click_ts, self.click_ts[pos_mask], self.click_ts[refund_mask]
        ])
        pay_ts = np.concatenate([
            self.pay_ts, self.pay_ts[pos_mask], self.pay_ts[refund_mask]
        ])
        refund_ts = np.concatenate([
            self.refund_ts, self.refund_ts[pos_mask], self.refund_ts[refund_mask]
        ])
        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)
        stream_pay_labels = copy.deepcopy(self.pay_labels)
        stream_pay_labels[pos_mask] = 0
        stream_pay_labels = np.concatenate([stream_pay_labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)
        stream_net_pay_labels = copy.deepcopy(self.net_pay_labels)
        stream_net_pay_labels[pos_mask] = 0
        stream_net_pay_labels = np.concatenate([stream_net_pay_labels, np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)
        stream_pay_mask = np.concatenate([np.ones_like(self.pay_labels),np.ones((np.sum(pos_mask),)), np.zeros((np.sum(refund_mask),))], axis=0)
        pay_labels = np.concatenate([pay_labels, pay_labels[pos_mask], pay_labels[refund_mask]],axis=0)
        net_pay_labels = np.concatenate([net_pay_labels, net_pay_labels[pos_mask], net_pay_labels[refund_mask]],axis=0)
        refund_labels = np.concatenate([refund_labels, refund_labels[pos_mask], refund_labels[refund_mask]],axis=0)
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
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_reddfm_duplicate_samples(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        ddfm退款版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        refund_mask = np.logical_and(self.refund_ts > 0,self.refund_labels)
        neg_dup_mask = np.logical_or(self.pay_ts < 0, self.pay_ts - self.click_ts > self.pay_attr_window)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        df2[neg_dup_mask],
        df3[refund_mask],
        ])
        
        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask],
            self.click_ts[neg_dup_mask] + self.pay_attr_window,
            self.refund_ts[refund_mask],
        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], 
            self.click_ts[~inw_pay_mask],
            self.click_ts[delay_pay_mask], 
            self.click_ts[neg_dup_mask],
            self.click_ts[refund_mask]], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], 
            self.pay_ts[~inw_pay_mask],
            self.pay_ts[delay_pay_mask],
            self.pay_ts[neg_dup_mask], 
            self.pay_ts[refund_mask]], axis=0)
        
        refund_ts = np.concatenate([
            self.refund_ts[inw_pay_mask], 
            self.refund_ts[~inw_pay_mask], 
            self.refund_ts[delay_pay_mask], 
            self.refund_ts[neg_dup_mask],
            self.refund_ts[refund_mask]], axis=0)


        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(delay_pay_mask),)),     
        np.zeros((np.sum(neg_dup_mask))),
        np.ones((np.sum(refund_mask))),     
        ], axis=0)


        stream_net_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.zeros((np.sum(~inw_pay_mask),)),   
        np.ones((np.sum(delay_pay_mask),)),  
        np.zeros((np.sum(neg_dup_mask),)),
        np.zeros((np.sum(refund_mask))),     
        ], axis=0)

        delay_pay_labels_afterPay = np.concatenate([
            np.zeros((np.sum(inw_pay_mask),)),
            np.zeros((np.sum(~inw_pay_mask),)),
            np.ones((np.sum(delay_pay_mask),)),
            np.ones((np.sum(neg_dup_mask),)),
            np.zeros((np.sum(refund_mask))),
        ], axis=0)

        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),
        np.ones((np.sum(neg_dup_mask),)),
        np.zeros((np.sum(refund_mask)))
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask], 
        pay_labels[neg_dup_mask],
        pay_labels[refund_mask]
        ], axis=0)

        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask],  
        net_pay_labels[neg_dup_mask],
        net_pay_labels[refund_mask],     
        ], axis=0)

        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask], 
        refund_labels[neg_dup_mask],
        refund_labels[refund_mask],          
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
                      delay_pay_labels_afterPay=delay_pay_labels_afterPay[idx],
                      stream_pay_labels=stream_pay_labels[idx],
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_reddfm_duplicate_samples_wo_refund_backfill(self, pay_wait_window,refund_wait_window):
        """
        在观测窗口内增加重定义和重复样本,注意这里对于退款相关的假正样本的回补
        ddfm退款版本,我们要把观测窗口内的转化和净转化分开
        Args:
            pay_wait_window (int): 支付等待窗口的时间长度。
            refund_wait_window (int): 退款等待窗口的时间长度。

        Returns:
            DataDF: 包含新特征、点击时间、支付时间、采样时间、退款时间、支付标签、净支付标签、退款标签和流支付标签的数据结构。

        """

        inw_pay_mask = np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) <= pay_wait_window)
        inw_refund_mask =np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) <= refund_wait_window)
        inw_net_pay_mask = np.logical_and(inw_pay_mask, ~inw_refund_mask)
        delay_pay_mask = np.logical_and(np.logical_and(self.pay_ts > 0, (self.pay_ts - self.click_ts) > pay_wait_window),self.pay_labels)
        delay_refund_mask = np.logical_and(np.logical_and(self.refund_ts > 0, (self.refund_ts - self.pay_ts) > refund_wait_window),self.refund_labels)
        neg_dup_mask = np.logical_or(self.pay_ts < 0, self.pay_ts - self.click_ts > self.pay_attr_window)

        df1 = self.features.copy(deep=True)
        df2 = self.features.copy(deep=True)
        df3 = self.features.copy(deep=True)
        new_features = pd.concat([
        df1[inw_pay_mask],
        df1[~inw_pay_mask],
        df2[delay_pay_mask],
        df2[neg_dup_mask],
        ])
        

        sample_ts = np.concatenate([
            self.click_ts[inw_pay_mask] + pay_wait_window,
            self.click_ts[~inw_pay_mask] + pay_wait_window,
            self.pay_ts[delay_pay_mask],
            self.click_ts[neg_dup_mask] + self.pay_attr_window,

        ])
        
        click_ts = np.concatenate([
            self.click_ts[inw_pay_mask], 
            self.click_ts[~inw_pay_mask],
            self.click_ts[delay_pay_mask], 
            self.click_ts[neg_dup_mask],
            ], axis=0)
        pay_ts = np.concatenate([
            self.pay_ts[inw_pay_mask], 
            self.pay_ts[~inw_pay_mask],
            self.pay_ts[delay_pay_mask],
            self.pay_ts[neg_dup_mask], 
            ], axis=0)
        
        refund_ts = np.concatenate([
            self.refund_ts[inw_pay_mask], 
            self.refund_ts[~inw_pay_mask], 
            self.refund_ts[delay_pay_mask], 
            self.refund_ts[neg_dup_mask],
            ], axis=0)


        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)

        stream_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),      
        np.zeros((np.sum(~inw_pay_mask),)),     
        np.ones((np.sum(delay_pay_mask),)),     
        np.zeros((np.sum(neg_dup_mask))),
        ], axis=0)

        stream_net_pay_labels = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),   
        np.zeros((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),  
        np.zeros((np.sum(neg_dup_mask),)),
        ], axis=0)


        delay_pay_labels_afterPay = np.concatenate([
            np.zeros((np.sum(inw_pay_mask),)),
            np.zeros((np.sum(~inw_pay_mask),)),
            np.ones((np.sum(delay_pay_mask),)),
            np.ones((np.sum(neg_dup_mask),)),
        ], axis=0)

        stream_pay_mask = np.concatenate([
        np.ones((np.sum(inw_pay_mask),)),  
        np.ones((np.sum(~inw_pay_mask),)),
        np.ones((np.sum(delay_pay_mask),)),
        np.ones((np.sum(neg_dup_mask),)),
        ], axis=0)
        
        pay_labels = np.concatenate([
        pay_labels[inw_pay_mask],               
        pay_labels[~inw_pay_mask],                
        pay_labels[delay_pay_mask], 
        pay_labels[neg_dup_mask],
        ], axis=0)


        net_pay_labels = np.concatenate([
        net_pay_labels[inw_pay_mask],          
        net_pay_labels[~inw_pay_mask],     
        net_pay_labels[delay_pay_mask],  
        net_pay_labels[neg_dup_mask],
        ], axis=0)


        refund_labels = np.concatenate([
        refund_labels[inw_pay_mask],            
        refund_labels[~inw_pay_mask],          
        refund_labels[delay_pay_mask], 
        refund_labels[neg_dup_mask],     
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
                      delay_pay_labels_afterPay=delay_pay_labels_afterPay[idx],
                      stream_pay_labels=stream_pay_labels[idx],
                      stream_net_pay_labels=stream_net_pay_labels[idx],
                      stream_pay_mask=stream_pay_mask[idx],
                      pay_attr_window=self.pay_attr_window,
                      refund_attr_window=self.refund_attr_window)

    def add_ddfm_duplicate_samples(self,pay_wait_window):
        """
        为数据集添加延迟支付的伪负样本。

        Args:
            pay_wait_window (float): 支付等待窗口时间，用于判断支付是否延迟。

        Returns:
            DataDF: 包含伪负样本的新数据集。
        """
        inw_pay_mask = np.logical_and((self.pay_ts - self.click_ts) <= pay_wait_window,self.pay_labels)
        delay_pay_mask = np.logical_and((self.pay_ts - self.click_ts) > pay_wait_window,self.pay_labels)
        neg_dup_mask = np.logical_or(self.pay_ts < 0, self.pay_ts - self.click_ts > self.pay_attr_window)
        new_features = pd.concat([
        self.features[inw_pay_mask],
        self.features[~inw_pay_mask],
        self.features[delay_pay_mask].copy(deep=True),
        self.features[neg_dup_mask].copy(deep=True)
        ])

        sample_ts = np.concatenate(
            [self.click_ts[inw_pay_mask]  + pay_wait_window,
             self.click_ts[~inw_pay_mask] + pay_wait_window,
             self.pay_ts[delay_pay_mask],
             self.click_ts[neg_dup_mask] + pay_wait_window], axis=0
        )

        click_ts = np.concatenate(
            [self.click_ts[inw_pay_mask], 
             self.click_ts[~inw_pay_mask],
             self.click_ts[delay_pay_mask],
             self.click_ts[neg_dup_mask]], axis=0
        )
        pay_ts = np.concatenate(
            [self.pay_ts[inw_pay_mask], 
             self.pay_ts[~inw_pay_mask],
             self.pay_ts[delay_pay_mask],
             self.pay_ts[neg_dup_mask]], axis=0
        )
        refund_ts = np.concatenate(
            [self.refund_ts[inw_pay_mask], 
             self.refund_ts[~inw_pay_mask],
             self.refund_ts[delay_pay_mask],
             self.refund_ts[neg_dup_mask]], axis=0
        )

        pay_labels = copy.deepcopy(self.pay_labels)
        net_pay_labels = copy.deepcopy(self.net_pay_labels)
        refund_labels = copy.deepcopy(self.refund_labels)


        stream_pay_labels = np.concatenate([
            np.ones((np.sum(inw_pay_mask),)),
            np.zeros((np.sum(~inw_pay_mask),)),
            np.ones((np.sum(delay_pay_mask),)),
            np.zeros((np.sum(neg_dup_mask),))
        ], axis=0)

        delay_pay_labels_afterPay = np.concatenate([
            np.zeros((np.sum(inw_pay_mask),)),
            np.zeros((np.sum(~inw_pay_mask),)),
            np.ones((np.sum(delay_pay_mask),)),
            np.ones((np.sum(neg_dup_mask),))
        ], axis=0)

        pay_labels = copy.deepcopy(self.pay_labels)
        pay_labels = np.concatenate([pay_labels[inw_pay_mask], 
                                     pay_labels[~inw_pay_mask],
                                     pay_labels[delay_pay_mask],
                                     pay_labels[neg_dup_mask]],axis=0)
        
        net_pay_labels = np.concatenate([net_pay_labels[inw_pay_mask], 
                                         net_pay_labels[~inw_pay_mask],
                                         net_pay_labels[delay_pay_mask],
                                         net_pay_labels[neg_dup_mask]],axis=0)
        
        refund_labels = np.concatenate([refund_labels[inw_pay_mask],
                                        refund_labels[~inw_pay_mask],
                                        refund_labels[delay_pay_mask],
                                        refund_labels[neg_dup_mask]],axis=0)


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
                      delay_pay_labels_afterPay=delay_pay_labels_afterPay[idx],
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
        if mode == "defer_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode == "base_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode == "defer_dp_pretrain":
            train_data = data_src.sub_days_v2(args.train_split_days_start, args.train_split_days_end,pay_wait_window).shuffle()
            test_data = data_src.sub_days_v2(args.test_split_days_start, args.test_split_days_end,pay_wait_window)
            data = {"train": train_data, "test": test_data}
        elif mode =="oracle_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.test_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode =="reoracle_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.test_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode =="vanilla_pay_inw_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode =="vanilla_netpay_inw_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode == "esdfm_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode == "esdfm2_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode == "esdfm_tn_dp_pretrain":
            train_data = data_src.sub_days_v2(args.train_split_days_start, args.train_split_days_end,pay_wait_window).shuffle()
            test_data = data_src.sub_days_v2(args.test_split_days_start, args.test_split_days_end,pay_wait_window)
            data = {"train": train_data, "test": test_data}
        elif mode == "esdfm2_inwpay_tn_pretrain":
            train_data = data_src.sub_days_inwspay_tn(args.train_split_days_start, args.train_split_days_end,pay_wait_window).shuffle()
            test_data = data_src.sub_days_inwspay_tn(args.test_split_days_start, args.test_split_days_end,pay_wait_window)
            data = {"train": train_data, "test": test_data}
        elif mode == "fnw_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode == "fnc_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode == "defuse_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
            data = {"train": train_data, "test": test_data}
        elif mode == "defuse_tn_dp_pretrain":
            train_data = data_src.sub_days_v2(args.train_split_days_start, args.train_split_days_end,pay_wait_window).shuffle()
            test_data = data_src.sub_days_v2(args.test_split_days_start, args.test_split_days_end,pay_wait_window)
            data = {"train": train_data, "test": test_data}
        elif mode == "bidefuse_pretrain":
            train_data = data_src.sub_days_v3(args.train_split_days_start, args.train_split_days_end,pay_wait_window).shuffle()
            test_data = data_src.sub_days_v3(args.test_split_days_start, args.test_split_days_end,pay_wait_window)
            data = {"train": train_data, "test": test_data}
        elif mode =="redefer_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="redefer_v1_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="redefer_v1_dp_dr_pretrain":
            train_data = data_src.sub_days_v4(args.train_split_days_start, args.train_split_days_end,pay_wait_window,refund_wait_window).shuffle()
            test_data = data_src.sub_days_v4(args.test_split_days_start, args.test_split_days_end,pay_wait_window,refund_wait_window)
        elif mode =="refnc_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="refnw_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "reesdfm_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "redefuse_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "reddfm_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="ddfm_pretrain":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end).shuffle()
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "ddfm_tn_dp_pretrain":
            train_data = data_src.sub_days_v2(args.train_split_days_start, args.train_split_days_end,pay_wait_window).shuffle()
            test_data = data_src.sub_days_v2(args.test_split_days_start, args.test_split_days_end,pay_wait_window)
            data = {"train": train_data, "test": test_data}
        else:
            raise ValueError(f"Unknown mode {mode}")
        print("writing data to cache file")
        if cache_path != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_data, "test": test_data}, f)
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


    if hasattr(test_data, 'features'):
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
    else:
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
            "test": test_data
        }

def get_ali_dataset_stream(args):
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
        train_stream = data["train"]
        test_stream = data["test"]
    
    else:
        print("building datasets")
        df,click_ts,pay_ts,refund_ts = get_data_df(args)
        pay_wait_window = args.pay_wait_window * SECONDS_A_DAY
        refund_wait_window = args.refund_wait_window * SECONDS_A_DAY
        pay_attr_window = args.pay_attr_window * SECONDS_A_DAY
        refund_attr_window = args.refund_attr_window * SECONDS_A_DAY
        data_src = DataDF(features = df, click_ts=click_ts, pay_ts=pay_ts, refund_ts=refund_ts,pay_attr_window=pay_attr_window,refund_attr_window=refund_attr_window)

        train_stream = []
        test_stream = []
        print("splitting into train and test sets")
        if mode == "defer_train_stream":
            train_data= data_src.sub_days(0, args.train_split_days_end).add_defer_duplicate_samples(pay_wait_window=pay_wait_window)
            train_data= train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "oracle_train_stream":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "reoracle_train_stream":
            train_data = data_src.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "vanilla_pay_inw_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_vanilla_pay_inw_samples(pay_wait_window=pay_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end) 
        elif mode == "vanilla_netpay_inw_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_vanilla_netpay_inw_samples(pay_wait_window=pay_wait_window, refund_wait_window = refund_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "esdfm_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_esdfm_fake_neg(pay_wait_window = pay_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "esdfm2_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_esdfm_fake_neg(pay_wait_window = pay_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "fnw_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_fnw_fake_neg()
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "fnc_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_fnc_fake_neg()
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "base_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_base_duplicate_samples()
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "base_train_stream_for_preliminary_experiment":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_base_duplicate_samples_for_Preliminary_Experiment()
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "defuse_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_defuse_fake_neg(pay_wait_window = pay_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "bidefuse_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_bidefuse_inw_outw_delay_postitive(pay_wait_window = pay_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="redefer_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_redefer_duplicate_samples(pay_wait_window = pay_wait_window,refund_wait_window = refund_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="redefer_train_stream_wo_pos_backfill":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_redefer_duplicate_samples_v1(pay_wait_window = pay_wait_window,refund_wait_window = refund_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="redefer_v1_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_redefer_v1_duplicate_samples(pay_wait_window = pay_wait_window,refund_wait_window = refund_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="redefer_v2_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_redefer_v2_duplicate_samples(pay_wait_window = pay_wait_window,refund_wait_window = refund_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="refnc_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_refnc_duplicate_samples()
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="refnw_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_refnw_duplicate_samples()
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="reesdfm_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_reesdfm_duplicate_samples(pay_wait_window = pay_wait_window,refund_wait_window = refund_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode =="redefuse_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_redefuse_duplicate_samples(pay_wait_window = pay_wait_window,refund_wait_window = refund_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)
        elif mode == "reddfm_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_reddfm_duplicate_samples(pay_wait_window = pay_wait_window,refund_wait_window = refund_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)       
        elif mode == "ddfm_train_stream":
            train_data = data_src.sub_days(0, args.train_split_days_end).add_ddfm_duplicate_samples(pay_wait_window = pay_wait_window)
            train_data = train_data.sub_days(args.train_split_days_start, args.train_split_days_end)
            test_data = data_src.sub_days(args.test_split_days_start, args.test_split_days_end)        
        else:
            raise ValueError(f"Unknown mode {mode}")
        for i in np.arange(args.train_split_days_start,args.train_split_days_end,args.stream_wait_window):
            train_day = train_data.sub_days(i,i+args.stream_wait_window)
            train_stream.append({"features": train_day.features,
                                    "click_ts": train_day.click_ts,
                                    "pay_ts": train_day.pay_ts,
                                    "sample_ts": train_day.sample_ts,
                                    "refund_ts": train_day.refund_ts,
                                    "pay_labels": train_day.pay_labels,
                                    "net_pay_labels": train_day.net_pay_labels,
                                    "refund_labels": train_day.refund_labels,
                                    "delay_pay_labels_afterPay" : train_day.delay_pay_labels_afterPay,
                                    "delay_pay_label_afterRefund" : train_day.delay_pay_label_afterRefund,
                                    "inw_pay_labels_afterPay" : train_day.inw_pay_labels_afterPay,
                                    "delay_refund_label_afterRefund" : train_data.delay_refund_label_afterRefund,
                                    "inw_pay_labels_afterRefund" : train_day.inw_pay_labels_afterRefund,
                                    "stream_pay_labels": train_day.stream_pay_labels,
                                    "stream_net_pay_labels": train_day.stream_net_pay_labels,
                                    "stream_pay_mask": train_day.stream_pay_mask})
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
                pickle.dump({"train": train_stream, "test": test_stream}, f)

    print("====== Train SET ======")
    for day in range(len(train_stream)):
        print("Day",day)
        print(f"Total samples                : {len(train_stream[day]['pay_labels']):,}")
        print(f"Positive pay labels         : {sum(train_stream[day]['pay_labels']):,}")
        print(f"Positive net pay labels      : {sum(train_stream[day]['net_pay_labels']):,}")
        print(f"Positive refund labels       : {sum(train_stream[day]['refund_labels']):,}")
    print("====== Test SET ======")
    for day in range(len(test_stream)):
        print("Day",day)
        print(f"Total samples                : {len(test_stream[day]['pay_labels']):,}")
        print(f"Positive pay labels         : {sum(test_stream[day]['pay_labels']):,}")
        print(f"Positive net pay labels      : {sum(test_stream[day]['net_pay_labels']):,}")
        print(f"Positive refund labels       : {sum(test_stream[day]['refund_labels']):,}")

    return {"train": train_stream,"test": test_stream}
            

