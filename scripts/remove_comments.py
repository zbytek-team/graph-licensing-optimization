import tokenize 
from io import StringIO 
from pathlib import Path 
import sys 


def strip_comments (src :str )->str :
    tokens =tokenize .generate_tokens (StringIO (src ).readline )
    out =[]
    for toknum ,tokval ,_ ,_ ,_ in tokens :
        if toknum ==tokenize .COMMENT :
            continue 
        out .append ((toknum ,tokval ))
    return tokenize .untokenize (out )


def process_file (path :Path )->None :
    text =path .read_text (encoding ="utf-8")
    new =strip_comments (text )
    if new !=text :
        path .write_text (new ,encoding ="utf-8")


def main (root :Path )->int :
    for p in root .rglob ("*.py"):
        if any (part .startswith (".")for part in p .parts ):
            continue 
        if "__pycache__"in p .parts :
            continue 
        if ".venv"in p .parts or "venv"in p .parts :
            continue 
        if ".git"in p .parts :
            continue 
        process_file (p )
    return 0 


if __name__ =="__main__":
    root =Path (".")
    sys .exit (main (root ))
