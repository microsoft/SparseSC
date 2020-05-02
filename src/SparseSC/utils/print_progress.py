# -*- coding: utf-8 -*-
""" A utility for displaying a progress bar
"""
# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a 
import sys
import datetime
SparseSC_prev_iteration = 0
def print_progress(iteration, total=100, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create progress bar. If this isn't a tty-like (re-writable) output then 
    no percent-complete number will be outputted.
    @params:
    iteration   - Required  : current iteration (Int)
    total       - Required  : total iterations (Int)
    prefix      - Optional  : prefix string (Str)
    suffix      - Optional  : suffix string (Str)
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    bar_length  - Optional  : character length of bar (Int)
    """
    fill_char = '>' # Note that the "█" character is not compatible with every platform...
    filled_length = int(round(bar_length * iteration / float(total)))
    if sys.stdout.isatty():
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        
        progress_bar = fill_char * filled_length + '-' * (bar_length - filled_length) 

        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, progress_bar, percents, '%', suffix))

        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()
    else: # Can't do interactive re-writing (e.g. w/ the /r character)
        global SparseSC_prev_iteration
        if iteration == 0:
            sys.stdout.write('%s |' % (prefix))
        else:
            if iteration <= SparseSC_prev_iteration:
                SparseSC_prev_iteration = 0
            prev_fill_length = int(round(bar_length * SparseSC_prev_iteration / float(total)))
            progress_bar = fill_char * (filled_length-prev_fill_length)

            sys.stdout.write('%s' % (progress_bar))

            if iteration == total:
                sys.stdout.write('| %s\n' % (suffix))
        SparseSC_prev_iteration = iteration
        sys.stdout.flush()

def log_if_necessary(str_note_start, verbose):
    str_note = str_note_start + ": " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if verbose>0:
        print(str_note)
    if verbose>1:
        print_memory_snapshot(extra_str=str_note)


def print_memory_snapshot(extra_str=None):
    import os
    import tracemalloc
    log_file = os.getenv("SparseSC_log_file") #None if non-existant
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    if log_file is not None:
        log_file = open(log_file, "a")
    if extra_str is not None:
        print(extra_str, file=log_file)
    limit=10
    print("[ Top 10 ] ", file=log_file)
    for stat in top_stats[:limit]:
        print(stat, file=log_file)
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024), file=log_file)
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024), file=log_file)
    #if old_snapshot is not None:
    #    diff_stats = snapshot.compare_to(old_snapshot, 'lineno')
    if log_file is not None:
        log_file.close()
