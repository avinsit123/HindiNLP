#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:33:13 2020

@author: r17935avinash
"""

from HindiNLPTools.HindiNer import NER
from HindiNLPTools.AutoClassifier import classifier


if __name__ == "__main__":
    detect_ner = NER()
    sentence = detect_ner.Predict("अविनाश आगरा में रहता है")
    print(sentence)    
