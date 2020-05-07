
## To run the project
Before compiling, modules should be loaded as follows:
```bash
module load gcc/6.2.0
module load openmpi/3.0-gcc-6.2
module load cuda/10.0
```

For GPU version, allocation can be done as follows:
```bash
salloc --nodes=1 --mem=1000000 --gres=gpu:K420:1 -t 00:05:00 -A edu19.DD2360
```

To run the project:
```bash
srun -n 1 nvprof ./bin/sputniPIC.out inputfiles/GEM_2D.inp
```
