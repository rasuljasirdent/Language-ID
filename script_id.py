# -*- coding: utf-8 -*-
"""
Rasul Dent
LIN 3012
This file identifies characters by unicode block
"""
from collections import namedtuple, Counter
def char_range(start, char, stop):
    return start <=char and char <= stop

def european_latin(char):
    #Basic Latin 
    return char_range("\u0041", char, "\u00FF")

def latin_a(char):
    return char_range("\u0100", char, "\u017F")

def latin_b(char):
    #Exclude basic Latin when clustering but combine when doing analysis
    return char_range("\u0180", char, "\u024F")

def extended_latin(char):
    return latin_a(char) or latin_b(char)
    
def greek(char):
    return char_range("\u0370", char, "\u03FF")

def cyrillic(char):
    return char_range("\u0400", char, "\u052F")

def arabic(char):
    #Arabic, Arabic supplement and Arabic extended
    return char_range("\u0600", char, "\u08FE")

def devanagari(char):
    return char_range("\u0900", char, "\u097F")

def bengali(char):
    return char_range("\u0980", char, "\u09FF")

def punjabi(char):
    return char_range("\u0A00", char, "\u0A7F")

def gujarati(char):
    return char_range("\u0A80", char, "\u0AFF")

def tamil(char):
    return char_range("\u0B80", char, "\u0BFF")

def malayam(char):
    return char_range("\u0D00", char, "\u0D7F")

def thai(char):
    return char_range("\u0E00", char, "\u0E7F")

def japanese(char):
    #hiragana or katakana
    return char_range("\u3040", char, "\u30FF")

def hangul(char):
    #korean characters are split across unicode
    return char_range("\u1100", char, "\u11FF") or char_range("\uAC00)", char, "\uD7AF")

def unified_cjk(char):
    return char_range("\u4E00", char, '\u9FFF')

def east_asian(char):
    return unified_cjk(char) or japanese(char) or hangul(char)

def one_of(char, scripts):
    return sum([script(char) for script in scripts])

def other(char):
    return True

named_scripts = [european_latin, extended_latin, greek, cyrillic,
                 arabic, devanagari, bengali, punjabi, tamil, malayam,
                 thai, japanese, hangul, unified_cjk, other]
labels = ["lat-eu", "lat-ex", "grk", "cyr",
          "arb", "dvn", "bng", 
          "pnj", "tml", "mym", 
          "thai", "jpn", "hgl",
          "cjk", "other"]
default_langs = ["en", "vi", "el", "ru", 
                 "ar", "hi", "bn",
                 "pa", "ta","my", 
                 "th", "ja", "ko",
                 "zh-tw", "am"]

script_defaults = dict(zip(labels, default_langs))

Script = namedtuple("Script", ["test", "label"])
labelled_scripts = [Script(test, label) for test, label in zip(named_scripts,labels)]

def classify_by_script(string, scripts=labelled_scripts):
    counter = Counter()
    for char in string:
        for script in scripts:
            if script.test(char):
                counter[script.label] +=1
                break
    top = counter.most_common(2)
    label = "other"
    if len(top) > 0:
        label = top[0]
    return (label, counter)
