from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(10, dtype='i')
    print('processor', rank, 'before bcast:', data)
else:
    data = np.empty(10, dtype='i')
    #print('processor', rank, 'before bcast:', data)
comm.Bcast(data, root=0)
for i in range(10):
    assert data[i] == i
print('processor', rank, 'after bcast:', data)
