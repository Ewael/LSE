#!/usr/bin/env python3

from AES_128_ECB import *
import binascii
import difflib

def floydRadius(f, x0, distance, radius, debug=False):
    """
    x0 -> f(x0) -> f(f(x0)) -> ... -> x -> ...

    when does x ~= x0?
    => when does distance(x0, x) > radius?

    where distance measures:
    - number of different bits      OR
    - numbers of differents chars   OR
    - length of longest common substring // ! change inequality to `<=`

    return m such as x_m == x_2m
    """
    if debug:
        print(f"[+] floyd_radius with x0 = {x0}, radius = {radius}")
    index = 1
    tortoise = f(x0)
    hare = f(f(x0))
    while distance(tortoise, hare) <= radius:
        index += 1
        tortoise = f(tortoise) # one step
        hare = f(f(hare)) # two steps
    return index

def charCommon(a, b, debug=False): # distance function
    """
    return how many common chars `a` and `b` have
    """
    assert len(a) == len(b), f"String have different length: len(a) = {len(a)}, len(b) = {len(b)}"
    count = 0
    z = zip(a, b)
    for i, j in z:
        if i == j:
            count += 1
    return count

def bitCommon(a, b, debug=False): # distance function
    """
    return how many common bits `a` and `b` have
    """
    assert len(a) == len(b), f"String have different length: len(a) = {len(a)}, len(b) = {len(b)}"
    nb_blocks = len(a) // 16
    # we keep leading zeroes when converting to bin
    bit_a = bin(int(binascii.hexlify(a), 16))[2:].zfill(128 * nb_blocks)
    bit_b = bin(int(binascii.hexlify(b), 16))[2:].zfill(128 * nb_blocks)
    bit_common = charCommon(bit_a, bit_b)
    if debug:
        print(f"[+] bit_a = {bit_a}")
        print(f"[+] bit_b = {bit_b}")
        print(f"[+] bit_common = {bit_common}")
    return bit_common

def longestCommonSubstring(a, b, debug=False):
    """
    return longest common substring between a and b
    """
    assert len(a) == len(b), f"String have different length: len(a) = {len(a)}, len(b) = {len(b)}"
    match = difflib.SequenceMatcher(None, a, b)
    lcs = match.find_longest_match(0, len(a), 0, len(b))
    if debug:
        print(f"[+] returned object = {lcs}")
        print(f"[+] lcs = '{a[lcs.a: lcs.a + lcs.size]}'")
    return lcs

def bitDistance(a, b, debug=False):
    """
    return length of longest common substring between binary representations of a and b
    """
    assert len(a) == len(b), f"String have different length: len(a) = {len(a)}, len(b) = {len(b)}"
    nb_blocks = len(a) // 16
    # we keep leading zeroes when converting to bin
    bit_a = bin(int(binascii.hexlify(a), 16))[2:].zfill(128 * nb_blocks)
    bit_b = bin(int(binascii.hexlify(b), 16))[2:].zfill(128 * nb_blocks)
    bit_lcs = longestCommonSubstring(a, b, debug)
    return bit_lcs.size

def charDistance(a, b, debug=False):
    lcs = longestCommonSubstring(a, b, debug)
    return lcs.size

print(bitDistance(b"thomas is bg", b"thomas is ko", True))
print(charDistance(b"thomas is bg", b"thomas is ko", True))
