import numpy as np

def exportmatches(matches):
    with open("matches","w") as f:
        for i in matches:
            #for j in i:
            #    f.write(str(j))
            #    f.write(" ")
            #f.write("\n")
            f.write(str(i))
            f.write("\n")
        f.close()
        