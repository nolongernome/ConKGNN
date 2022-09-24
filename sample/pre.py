import os
path=os.getcwd()
for filedir in os.listdir(path):
    filedir_path = os.path.join(path, filedir)
    newname=filedir_path.replace("R97","Drugs")
    os.rename(os.path.join(path, filedir_path), os.path.join(path, newname))