#!/usr/bin/python
"""
    Copy from https://github.com/kylerbrown/textgrid/blob/master/textgrid.py
"""

from collections import namedtuple

Entry = namedtuple("Entry", ["start",
                             "stop",
                             "name",
                             "tier"])

def read_textgrid(filename, fileEncoding="utf-8"):
    """
    Reads a TextGrid file into a dictionary object
    each dictionary has the following keys:
    "start"
    "stop"
    "name"
    "tier"

    Points and intervals use the same format, 
    but the value for "start" and "stop" are the same

    Optionally, supply fileEncoding as argument. This defaults to "utf-8", tested with 'utf-16-be'.
    """
    if isinstance(filename, str):
        with open(filename, "r", encoding=fileEncoding) as f:
            content = _read(f)
    elif hasattr(filename, "readlines"):
        content = _read(filename)
    else:
        raise TypeError("filename must be a string or a readable buffer")

    interval_lines = [i for i, line in enumerate(content)
                      if line.startswith("intervals [")
                      or line.startswith("points [")]
#    tier_lines, tiers =  [(i, line.split('"')[-2]) 
#            for i, line in enumerate(content)
#            if line.startswith("name =")]
    tier_lines = []
    tiers = []
    for i, line in enumerate(content):
        if line.startswith("name ="):
            tier_lines.append(i)
            tiers.append(line.split('"')[-2]) 

    interval_tiers =  _find_tiers(interval_lines, tier_lines, tiers)
    assert len(interval_lines) == len(interval_tiers)
    return [_build_entry(i, content, t) for i, t in zip(interval_lines, interval_tiers)]


def _find_tiers(interval_lines, tier_lines, tiers):
    tier_pairs = zip(tier_lines, tiers)
    cur_tline, cur_tier = next(tier_pairs) 
    next_tline, next_tier = next(tier_pairs, (None, None))
    tiers = []
    for il in interval_lines:
        if next_tline is not None and il > next_tline:
            cur_tline, cur_tier = next_tline, next_tier
            next_tline, next_tier = next(tier_pairs, (None, None))           
        tiers.append(cur_tier)
    return tiers 


def _read(f):
    return [x.strip() for x in f.readlines()]

def write_csv(textgrid_list, filename=None, sep=",", header=True, save_gaps=False, meta=True):
    """
    Writes a list of textgrid dictionaries to a csv file.
    If no filename is specified, csv is printed to standard out.
    """
    columns = list(Entry._fields)
    if filename:
        f = open(filename, "w")
    if header:
        hline = sep.join(columns)
        if filename:
            f.write(hline + "\n")
        else:
            print(hline)
    for entry in textgrid_list:
        if entry.name or save_gaps:  # skip unlabeled intervals
            row = sep.join(str(x) for x in list(entry))
            if filename:
                f.write(row + "\n")
            else:
                print(row)
    if filename:
        f.flush()
        f.close()
    if meta:
        with open(filename + ".meta", "w") as metaf:
            metaf.write("""---\nunits: s\ndatatype: 1002\n""")
        
def _build_entry(i, content, tier):
    """
    takes the ith line that begin an interval and returns
    a dictionary of values
    """
    start = _get_float_val(content[i + 1])  # addition is cheap typechecking
    if content[i].startswith("intervals ["):
        offset = 1
    else:
        offset = 0 # for "point" objects
    stop = _get_float_val(content[i + 1 + offset])
    label = _get_str_val(content[i + 2 + offset])
    return Entry(start=start, stop=stop, name=label, tier=tier)


def _get_float_val(string):
    """
    returns the last word in a string as a float
    """
    return float(string.split()[-1])


def _get_str_val(string):
    """
    returns the last item in quotes from a string
    """
    return string.split('"')[-2]


def textgrid2csv():
    import argparse
    parser = argparse.ArgumentParser(description="convert a TextGrid file to a CSV.")
    parser.add_argument("TextGrid",
                        help="a TextGrid file to process")
    parser.add_argument("-o", "--output", help="(optional) outputfile")
    parser.add_argument("--sep", help="separator to use in CSV output",
                        default=",")
    parser.add_argument("--noheader", help="no header for the CSV",
                        action="store_false")
    parser.add_argument("--savegaps", help="preserves intervals with no label",
            action="store_true")
    args = parser.parse_args()
    tgrid = read_textgrid(args.TextGrid)
    write_csv(tgrid, args.output, args.sep, args.noheader, args.savegaps)


if __name__ == "__main__":
    textgrid2csv()
