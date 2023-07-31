import os
from send_email import *
import timeit
from attack_utils import mem_stats
import subprocess
"""
Utility script to run MoPe or LOSS, over sizes and other parameters. 

This is a disk-space-saver since it deletes the perturbed MoPe models after a run is finished,
and saves all of the results to one central directory in the machine.
"""

attack_type = 'DetectGPT'
sizes = [ "70m", "160m", "410m", "1b", "1.4b", "2.8b"]
points = 2000
seed = 1930
email_flag = True
dir_name = attack_type

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # .communicate() will wait for the process to finish before moving to the next command
    output, error = process.communicate()
    if process.returncode != 0:
        print(f'Error occurred: {error.decode()}')
    else:
        print(f'Output: {output.decode()}')

def directory_exists(directory_path):
    return os.path.exists(directory_path) and os.path.isdir(directory_path)

if attack_type == 'MoPe':
    # MoPe version

    for s in sizes:
        os.mkdir(f"{s}")
        os.chdir(f"{s}")
        os.system("cp ../../*py .")
        name = f"{s}_result"
        cmd = f"python3 run_mope.py --mod_size {s} --n_models 30 --sigma 0.005 --model_half --pack --n_samples {points} --checkpoint step98000 --deduped --seed {seed} &>> out.txt"
        os.system(cmd)
        print(cmd)

        os.chdir("MoPe")
        os.system("rm -r pythia-*")
        os.mkdir(f"{s}_results")
        os.system(f"cp *png {s}_results")
        os.system(f"cp *html {s}_results")
        os.system(f"cp *pt {s}_results")
        os.chdir("../..")

        os.system(f"cp -r {s}/MoPe/{s}_results results")

        # os.system(f"cp -r {s}/LOSS/{s}_results results")
if attack_type == 'Loss':
    for s in sizes:
        os.mkdir(f"{s}")
        os.chdir(f"{s}")
        os.system("cp ../../*py .")
        name = f"{s}_result"
        cmd = f"python3 run_loss.py --mod_size {s} --model_half --pack --n_samples {points} --checkpoint step98000 --deduped --seed {seed} &>> out.txt"
        os.system(cmd)

        print(cmd)

        os.chdir("LOSS")
        os.mkdir(f"{s}_results")
        os.system(f"cp *png {s}_results")
        os.system(f"cp *pt {s}_results")
        os.chdir("../..")
        os.system(f"cp -r {s}/LOSS/{s}_results results")

elif attack_type == 'DetectGPT': 
    if email_flag: 
        email = str(input("Enter gmail: "))
        password = str(input("Enter App Password (not different to email password, google it): "))
        send_email_message(email, password, "Starting Run on Cluster", body='')
        run_start = timeit.default_timer()
    for s in sizes:
        if not directory_exists(dir_name): 
            os.mkdir(dir_name)
        
        cmd = f"python3 run_detectgpt.py --mod_size {s} --model_half --pack --n_samples {points} --checkpoint step98000 --deduped --seed {seed} --n_perts 5 "
        start = timeit.default_timer()
        print(cmd)
        run_command(cmd)
        end = timeit.default_timer()
        
        if email_flag:
            mem_string = mem_stats(return_string=True)
            send_email_message(email, password, f"Run Update: Size {s} Finished", body=f"cmd run: {cmd} \n time elapsed: {start-end}. /n {mem_string}", attach_dir=None)
    if email_flag: 
        run_end = timeit.default_timer()
        send_email_message(email, password, "URGENT: Shut down instance", body=f'Cluster run finished in {run_start-run_end} seconds. The command run was {cmd} over {points} points, and {sizes} sized models.')
            



