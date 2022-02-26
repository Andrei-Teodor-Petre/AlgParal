from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

s=rank

if rank > 0:
    src=rank-1
else:
    src=size-1    

if rank != size-1:
    dst=rank+1
else:
    dst=0    


comm.send(s,dest=dst)
m=comm.recv(source=src)
print('processor', rank, 'received a msg from processor', m)
