import sys

import numpy as np

job_id = sys.argv[1]
n_jobs = sys.argv[2]
time = 0

f_name = f"slurm-{job_id}_"
times = []
for n in range(int(n_jobs)):
    f = open(f_name + f"{n}" + ".out")
    lines = f.readlines()
    time_data = float(lines[-1][8:])
    time += time_data
    times.append(time_data)
    f.close()
print("total time : ", time)
print("maximum time for one job : ", np.max(times))
