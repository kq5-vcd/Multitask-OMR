import os

dir_list = []
for parent, dirnames, filenames in os.walk('Data/Samples'):
    """
    for fn in filenames:
        if fn[:2] == '._':
            os.remove(os.path.join(parent, fn))
    """
    dir_list = dirnames
    break

f = open("Data/sample_list.txt", "w")
for package in dir_list:
    f.write(package + "\n")
f.close()
