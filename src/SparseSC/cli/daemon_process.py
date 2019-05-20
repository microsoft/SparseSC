"""
A modeul to run as a background process

pip install psutil

usage: 
    python -m SparseSC.cli.daemon_process start
    python -m SparseSC.cli.daemon_process stop
    python -m SparseSC.cli.daemon_process status
"""
# pylint: disable=multiple-imports

import sys, os, time, atexit, signal, json,psutil
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from .scgrad import grad_part, DAEMON_FIFO, DAEMON_PID, _BASENAMES

#-- DAEMON_FIFO = "/tmp/sc-daemon.fifo"
#-- DAEMON_PID = "/tmp/sc-gradient-daemon.pid"
#-- 
#-- _CONTAINER_OUTPUT_FILE = "output.yaml"  # Standard Output file
#-- _GRAD_COMMON_FILE = "common.yaml"
#-- _GRAD_PART_FILE = "part.yaml"
#-- 
#-- _BASENAMES = [_GRAD_COMMON_FILE, _GRAD_PART_FILE, _CONTAINER_OUTPUT_FILE]



pidfile, fifofile = DAEMON_PID, DAEMON_FIFO


def stop():
    """Stop the daemon."""

    # Get the pid from the pidfile
    try:
        with open(pidfile, "r") as pf:
            pid = int(pf.read().strip())
    except IOError:
        pid = None

    if not pid:
        message = "pidfile {0} does not exist. " + "Daemon not running?\n"
        sys.stderr.write(message.format(pidfile))
        return  # not an error in a restart

    # Try killing the daemon process
    try:
        while 1:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.1)
    except OSError as err:
        e = str(err.args)
        if e.find("No such process") > 0:
            if os.path.exists(pidfile):
                os.remove(pidfile)
            if os.path.exists(fifofile):
                os.remove(fifofile)
        else:
            print(str(err.args))
            sys.exit(1)


def run():
    """
    do work
    """
    while True:
        with open(DAEMON_FIFO, "r") as fifo:
            try:
                params = fifo.read()
                print("params: " + params)
                sys.stdout.flush()
                tmpdirname, return_fifo, k = json.loads(params)
                print(_BASENAMES)
                for file in os.listdir(tmpdirname):
                    print(file)
                common_file, part_file, out_file = [
                    os.path.join(tmpdirname, name) for name in _BASENAMES
                ]
                print([common_file, part_file, out_file, return_fifo, k])
                sys.stdout.flush()

                # LOAD IN THE INPUT FILES
                with open(common_file, "r") as fp:
                    common = load(fp, Loader=Loader)
                with open(part_file, "r") as fp:
                    part = load(fp, Loader=Loader)

                # DO THE WORK
                print("about to do work: ")
                sys.stdout.flush()
                grad = grad_part(common, part, int(k))
                print("did work: ")
                sys.stdout.flush()

                # DUMP THE RESULT TO THE OUTPUT FILE
                with open(out_file, "w") as fp:
                    fp.write(dump(grad, Dumper=Dumper))

            except Exception as err:  # pylint: disable=broad-except

                # SOMETHING WENT WRONG, RESPOND WITH A NON-ZERO
                try:
                    with open(return_fifo, "w") as rf:
                        rf.write("1")
                except:  # pylint: disable=bare-except
                    print("double failed...: ")
                    sys.stdout.flush()
                else:
                    print(
                        "failed with {}: {}",
                        err.__class__.__name__,
                        getattr(err, "message", "<>"),
                    )
                    sys.stdout.flush()

            else:
                # SEND THE SUCCESS RESPONSE
                print("success...: ")
                sys.stdout.flush()
                with open(return_fifo, "w") as rf:
                    rf.write("0")
                print("and wrote about it...: ")
                sys.stdout.flush()


def start():
    """
    start the process_deamon if it's not already started
    """

    #  IS THE DAEMON ALREADY RUNNING?
    try:
        with open(pidfile, "r") as pf:
            pid = int(pf.read().strip())
    except IOError:
        pid = None

    if pid:
        message = "pidfile {0} already exist. " + "Daemon already running?\n"
        sys.stderr.write(message.format(pidfile))
        sys.exit(1)


    def delpid():
        " clean up"
        try:
            os.remove(pidfile)
        except: 
            print("failed to remove pidfile"); sys.stdout.flush()
            raise RuntimeError("failed to remove pidfile")
        else:
            print("removed pidfile"); sys.stdout.flush()
        try:
            os.remove(fifofile)
        except: 
            print("failed to remove fifofile"); sys.stdout.flush()
            raise RuntimeError("failed to remove pidfile")
        else:
            print("removed fifofile"); sys.stdout.flush()
    atexit.register(delpid)

    os.mkfifo(fifofile)  # pylint: disable=no-member

    pid = str(os.getpid())
    with open(pidfile, "w+") as f:
        f.write(pid + "\n")

    workdir = os.getenv("AZ_BATCH_TASK_WORKING_DIR","/tmp")
        

    # """Start the daemon."""
    sys.stdout = open(os.path.join(workdir,"sc-ps-out.txt"),"a+")
    sys.stderr = open(os.path.join(workdir,"sc-ps-err.txt"),"a+")
    print("process started >>>>>>>>>>>"); sys.stdout.flush()
    run()


def status():
    """
    check the process_deamon status
    """

    if ARGS[0] == "status":
        if not os.path.exists(DAEMON_PID):
            print("Daemon not running")
            return
        with open(DAEMON_PID,'r') as fh:
            _pid = int(fh.read())
        
        if _pid in psutil.pids():
            print("daemon process (pid {}) is running".format(_pid))
        else:
            print("daemon process (pid {}) NOT is running".format(_pid))

def main():
    ARGS = sys.argv[1:]

    if not ARGS:
        print("no args provided")
    elif ARGS[0] == "start":
        start()
    elif ARGS[0] == "stop":
        stop()
    elif ARGS[0] == "status":
        status()
    else:
        print("unknown command '{}'".format(ARGS[0]))


if __name__ == "__main__":
    main()
