from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = 100*rank #date locale de trimis


if rank != 0:
    src = rank-1
else:
    src = size-1
    
if rank != size-1:
    dst = rank+1
else:
    dst = 0


if rank%2 == 0:    
    comm.send(data,dest=dst)
    data1 = comm.recv(source=src)
else:
    data1 = comm.recv(source=src)
    comm.send(data,dest=dst)
    
print(data1)
