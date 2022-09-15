import easygui as eg
import sys


def read_file():
    return eg.fileopenbox()


sys.exit(read_file())
