"""
Created on Sat Apr 17 20:17:31 2021

@author: Subhajit.Debnath
"""

import os
import shutil

#Function to create directory
def createDir(root_dir,dir_name):
    try:
        if not os.path.exists(os.path.join(str(root_dir),str(dir_name))):
            os.makedirs(os.path.join(str(root_dir),str(dir_name)))
            return os.path.join(str(root_dir),str(dir_name))
    except:
        print("Error while creating " + str(dir_name) + " directory.")
        
        
def deleteDir(root_dir,dir_name):
    try:
        shutil.rmtree(os.path.join(str(root_dir),str(dir_name)), ignore_errors=True)
    except OSError as e:
        print("Error: %s : %s" % (os.path.join(str(root_dir),str(dir_name)), e.strerror))
        