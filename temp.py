# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BILSTM-CRF
# @File     :temp
# @Date     :2021/7/11 15:51
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch

a = torch.randn(5,6,7)
b = torch.randn(5,6,1)
print(a*b)