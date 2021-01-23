#!/usr/bin/env python3

from Crypto.Cipher import AES

def encrypt(plain, key, debug=False):
    AES_ECB = AES.new(key, AES.MODE_ECB)
    cipher = AES_ECB.encrypt(plain)
    if debug:
        print("[+] encrypting {} with key {} -> {}".format(plain, key, cipher))
    return cipher

def decrypt(cipher, key, debug=False):
    AES_ECB = AES.new(key, AES.MODE_ECB)
    plain = AES_ECB.decrypt(cipher)
    if debug:
        print("[+] decrypting {} with key {} -> {}".format(cipher, key, plain))
    return plain
