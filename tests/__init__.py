import subprocess
import sys
import os


def setup_package():
    fp = os.path.dirname(__file__)
    cmd = 'ln -s data %s' % fp
    subprocess.call(cmd.split())
