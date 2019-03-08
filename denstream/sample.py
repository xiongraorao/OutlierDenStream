#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:48:21 2017

@author: anr.putina
"""

class Sample():
    
    def __init__(self, value, timestamp=0):
        self.value = value
        self.timestamp = 0
        self.realTimestamp = timestamp
        
    def getValue(self):
        return self.value
    
    def setTimestamp(self, timestamp):
        self.timestamp = timestamp
        
    def setRealTimestamp(self, timestamp):
        self.realTimestamp = timestamp

    def setMicroClusterNumber(self, microClusterNumber):
        self.microClusterNumber = microClusterNumber