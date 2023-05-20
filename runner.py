import os

sizes = ["1.4B", "2.8B"]
perturb_amts = [0.001, 0.005, 0.01, 0.05]
N = 15

for size in sizes:
    for p in perturb_amts:
        dirname = f"{size}_{p}_{N}"
        os.mkdir(dirname)
        os.chdir(dirname)
        os.system("cp ../*py .")
        os.system(f"python run_mope.py --mod_size {size} --n_models {N} --n_samples 1000 --sigma {p}")
        os.chdir("..")
