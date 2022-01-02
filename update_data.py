import os
import random

def write_list(filename, shuffle=False):
    dir_list = []
    for parent, dirnames, filenames in os.walk('Data/' + filename):
        """
        for fn in filenames:
            if fn[:2] == '._':
                os.remove(os.path.join(parent, fn))
        """
        dir_list = dirnames
        break

    if shuffle:
        random.shuffle(dir_list) 

    f = open("Data/" + filename + "_list.txt", "w")
    for package in dir_list:
        f.write(package + "\n")
    f.close()

write_list('Corpus', True)