from mpi4py import MPI

print(MPI.COMM_WORLD) # ensure MPI works ?
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
	print('this is a grid')
	pass
	
else:
	print('this is a piece of the board')
	pass



config = {
    'size' : 10,
    'ocean' : '~',
    'shark' : 'X',
    'fish' : '0'
}




class Grid:

    def __init__(self, config):
        self.size = config['size']
        self.ocean = config['ocean']
        self.fish = config['fish']
        self.shark = config['shark']
        self.resetBoard()

    def show(self):
        for row in self.board:
            print(row)

    def resetBoard(self):
        row = [self.ocean for _ in range(0, self.size)]
        self.board = [row for _ in range(0, self.size)]

    def updateBoard(self, rules):
        raise NotImplementedError()


grid = Grid(config)
grid.show()

