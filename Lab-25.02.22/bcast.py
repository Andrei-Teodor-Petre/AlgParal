from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'key1' : [3, 24.62, 9+4j],
            'key2' : ( 'fmi', 'unibuc')}
else:
    data = None
data = comm.bcast(data, root=0)
print('processor', rank, 'receives')
print('\t', data)
