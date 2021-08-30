import subprocess
import os


def launch():
    os.chdir(r"H:\Study\Summer Prog 2021\\Tasks\Summer_Prog_2021\\Computer-Vision\\Face Detection\\aws-tf")
    subprocess.run("terraform apply --auto-approve")
