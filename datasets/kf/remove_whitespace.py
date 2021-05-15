import glob
import pandas as pd
import os

filenames = glob.glob("./*txt")

for filename in filenames:
    f = open(filename, 'r')
    content = f.read()
    content = content.replace(" ", '')
    f.close()

    f = open(f"./ns/{os.path.basename(filename)}", "w")
    f.write(content)
    f.close()
