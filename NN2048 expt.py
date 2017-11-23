#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 11:12:23 2017

@author: jojo
"""

from functools import reduce
def to_bits(*l):
    return reduce(lambda r,i: r |(1<<i), l, 0)

def winning_patterns():
    v1 = to_bits(0,1,2)
    h1 = to_bits(0,3,6)
    return [v1, v1<<3, v1<<6, h1, h1<<1, h1<<2, to_bits(0,4,8), to_bits(2,4,6)]


print(to_bits(0,1,2))
print(to_bits(0,3,6))
print(winning_patterns())