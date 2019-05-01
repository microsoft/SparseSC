with open("DifferentialTrends.py","r+") as f:
    new_f = f.readlines()
    f.seek(0)
    for line in new_f:
        if "get_ipython().magic" not in line:
            f.write(line)
    f.truncate()
