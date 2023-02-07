#! /usr/bin/python3

def palindrome(word):
    reverse = word[::-1]
    return word.lower() == reverse.lower()

def findem(minlength=4):
    for word in open("/local/dict/scowl.txt"):
        word = word.rstrip()    # remove newline at end
        if len(word) >= minlength and palindrome(word):
            print(word)

findem()
