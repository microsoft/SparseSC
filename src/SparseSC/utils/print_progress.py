# -*- coding: utf-8 -*-
""" A utility for displaying a progress bar
"""
# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a 
import sys
import datetime
SparseSC_prev_iteration = 0
def print_progress(iteration, total=100, prefix='', suffix='', decimals=1, bar_length=100, file=sys.stdout):
    """
    Call in a loop to create progress bar. If this isn't a tty-like (re-writable) output then 
    no percent-complete number will be outputted.
    @params:
    iteration   - Required  : current iteration (Int) (typically 1 is first possible)
    total       - Required  : total iterations (Int)
    prefix      - Optional  : prefix string (Str)
    suffix      - Optional  : suffix string (Str)
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    bar_length  - Optional  : character length of bar (Int)
    file        - Optional  : file descriptor for output
    """
    fill_char = '>' # Note that the "█" character is not compatible with every platform...
    empty_char = '-'
    filled_length = int(round(bar_length * iteration / float(total)))
    if file.isatty():
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        
        progress_bar = fill_char * filled_length + empty_char * (bar_length - filled_length) 

        file.write('\r%s |%s| %s%s %s' % (prefix, progress_bar, percents, '%', suffix))

        if iteration == total:
            file.write('\n')
        file.flush()
    else: # Can't do interactive re-writing (e.g. w/ the /r character)
        global SparseSC_prev_iteration
        if iteration == 1:
            file.write('%s |' % (prefix))
        else:
            if iteration <= SparseSC_prev_iteration:
                SparseSC_prev_iteration = 0
            prev_fill_length = int(round(bar_length * SparseSC_prev_iteration / float(total)))
            progress_bar = fill_char * (filled_length-prev_fill_length)

            file.write('%s' % (progress_bar))

            if iteration == total:
                file.write('| %s\n' % (suffix))
        SparseSC_prev_iteration = iteration
        file.flush()

def it_progressmsg(it, prefix="Loop", file=sys.stdout, count=None):
    for i, item in enumerate(it):
        if count is None:
            file.write(prefix + ": " + i + "\n")
        else:
            file.write(prefix + ": " + i + " of " + count + "\n")
        file.flush()
        yield item
    file.write(prefix + ": FINISHED\n")
    file.flush()

#Similar to above, but you wrap an iterator
def it_progressbar(it, prefix="", suffix='', decimals=1, bar_length=100, file=sys.stdout, count=None):
    if count is None:
        count = len(it)
    def show(j):
        fill_char = '>'
        empty_char = '-'
        x = int(bar_length*j/count)
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (j / float(count)))
        progress_bar = fill_char * x + empty_char * (bar_length - x) 
        if file.isatty():
            file.write("%s |%s| %s%s %s\r" % (prefix, progress_bar, percents, '%', suffix))
            file.flush()
        else: # Can't do interactive re-writing (e.g. w/ the /r character)
            if j == 0:
                file.write('%s |' % (prefix))
            else: 
                prev_x = int(bar_length*(j-1)/count)
                progress_bar = fill_char * (x-prev_x)
                file.write('%s' % (progress_bar))
                if j == count:
                    file.write('| %s' % (suffix))
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def log_if_necessary(str_note_start, verbose):
    str_note = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + str_note_start 
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
