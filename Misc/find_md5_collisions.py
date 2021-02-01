#!/usr/bin/env python3

import hashlib

string = b"b"
count = 1
while True:
    h = hashlib.md5(string).digest()
    if h[-2:] == b"AB":
        print(f"[x] hash = {hashlib.md5(string).hexdigest()}")
        print(f"[x] string = '{chr(string[0])}' * {count}")
        break
    string += b"a"
    count += 1
