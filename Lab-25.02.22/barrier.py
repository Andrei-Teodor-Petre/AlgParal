from mpi4py import MPI
from time import sleep
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank == 0:
    sleep(4)
    print("root has woken up")

comm.Barrier()

print('processor', rank, 'enters sleep ')
sleep(random.uniform(0, 10))
print('processor', rank, 'finished sleeping ')

comm.Barrier()

if rank == 0:
    print("All processors done!")
#print('processor', rank, 'done');
