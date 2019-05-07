# -*- coding: utf-8 -*-
""" A utility for displaying a progress bar
"""
# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a 
import sys
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
    iteration   - Required  : current iteration (Int)
    total       - Required  : total iterations (Int)
    prefix      - Optional  : prefix string (Str)
    suffix      - Optional  : suffix string (Str)
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    # Note that the "█" character is not compatible with every platform...
    progress_bar = '>' * filled_length + '-' * (bar_length - filled_length) 

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, progress_bar, percents, '%', suffix))

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
