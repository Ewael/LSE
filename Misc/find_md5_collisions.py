#!/usr/bin/env python3

from hashlib import md5
import random
import os

toFile = True
if toFile:
    os.system('rm -f logs.txt')

def log(msg):
    print(msg)
    if toFile == True:
        f = open("logs.txt", "a")
        f.write(msg + '\n')
        f.close()

def isGood(h):
    if h.digest()[-2:] == b"AB":
        return True
    return False

def findString():
    string = b"b"
    count = 1
    while True:
        h = md5(string)
        if isGood(h):
            log(f"[x] hash = {h.hexdigest()}, string = '{chr(string[0])}' * {count}")
            break
        string += bytes([string[0]])
        count += 1

def findNumber():
    number = 0
    while True:
        h = md5(str(number).encode('utf-8'))
        if isGood(h):
            log(f"[x] hash = {h.hexdigest()}, number = {number}")
            break
        number += 1

def findFile(filename):
    f = open(filename, errors='ignore')
    for line in f.readlines():
        l = line[:-1]
        h = md5(l.encode('utf-8'))
        if isGood(h):
            log(f"[x] hash = {h.hexdigest()}, string = '{l}'")

findString()
findNumber()
findFile('/usr/share/wordlists/rockyou.txt')
