# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 09:26:56 2022

@author: lenovo
"""
import os 
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
    '-p', 
    '--path'
    )
    
    args = parser.parse_args()
    
    with open(args.path, 'r') as f:
        print(f)
    
    


















