from enum import Enum
from typing import List, Tuple
import uuid
import math
from matplotlib import pyplot as plt

import numpy as np
from matplotlib import cm

from utils import Position, calculate_distance, reached_destination, AgentTypes


WIDTH, HEIGHT = 1000, 1000
POS_OFFSET = 50
BATCH_SIZE = 3

class Agent:
		
	def __init__(self, posi: Position, rank: int, type: AgentTypes):
		
		self.id = uuid.uuid4()

		self.rank = rank

		self.type = type

		self.position = posi
		self.destination = None

		self.max_velocity = 4
		self.acceleration = 0.1
		self.velocity_X = 0
		self.velocity_Y = 0
		self.alignment_error = 40

		self.external_factors = None

		self.current_age = 1

		self.cnt = 0

	def age(self):
		self.current_age += 1

	def can_breed(self) -> bool:
		return False
		if self.type is AgentTypes.SHARK:
			return False
		elif self.type is AgentTypes.FISH:
			return self.current_age >= 10 #and random.randint(0, 9) == 7

	def should_die(self) -> bool:
		return False
		if self.type is AgentTypes.SHARK:
			if self.fish_collision():
				self.current_age -= 10
			return self.current_age > 100

	def compile_forces(self):
		external_force = self.compile_external_force()
		gravity = self.compile_gravity(self.external_factors)
		return (external_force[0] + gravity[0]), (external_force[1] + gravity[1])

	def compile_external_force(self):
		return 3, 3

	def ensure_valid_position(self):
		if self.position.x >= WIDTH:
			self.position.x = WIDTH - 1
		if self.position.y >= HEIGHT:
			self.position.y = HEIGHT - 1

	def compile_gravity(self, external_factors: List[Tuple[AgentTypes, Position]]):
		
		x = np.arange(0,WIDTH,1)
		y = np.arange(0,HEIGHT,1)
		X, Y = np.meshgrid(x,y)

		ocean_cost_map = np.zeros(len(X))

		fish = [f[1] for f in external_factors if f[0] is AgentTypes.FISH]
		sharks = [s[1] for s in external_factors if s[0] is AgentTypes.SHARK]

		if len(fish) == 0 and len(sharks) == 0:
			return 0, 0

		for fish_dest in fish:
			ocean_cost_map = self.constructDestination(ocean_cost_map, X, Y, fish_dest)
		for shark_obstacle in sharks:
			ocean_cost_map = self.constructObstacle(ocean_cost_map, X, Y, shark_obstacle)
		
		dx,dy = np.gradient(ocean_cost_map,1,1)

		self.ensure_valid_position()
		pointI = Position(x[int(self.position.x)]-1, y[int(self.position.y)]-1)

		u = -dx[int(pointI.x),int(pointI.y)]
		v = -dy[int(pointI.x),int(pointI.y)]

		pointI.x = pointI.x + u
		pointI.y = pointI.y + v

		fish_arr = []
		for f in fish:
			fish_arr.append([f.x, f.y])

		sharks_arr = []
		for s in sharks:
			sharks_arr.append([s.x, s.y])

		

		obstacle = np.array([[10,14],[27,5],[31,17],[16,27]])

		atractiveCompArray = [0,0]
		for i in range(len(fish)):
			atractiveCompArray -= ([pointI.x,pointI.y] - np.array([fish[i].x,fish[i].y])) / calculate_distance(pointI, fish[i])
	
		repulsiveCompArray = [0,0]
		for i in range(len(sharks)):
			repulsiveCompArray += ([pointI.x,pointI.y] - np.array([sharks[i].x,sharks[i].y])) / calculate_distance(pointI, sharks[i])

		mvm = -(atractiveCompArray + repulsiveCompArray)

		#plt.ioff()

		# Make the plot
		# fig = plt.figure()
		# ax = fig.gca(projection='3d')
		# ax.plot_surface(Y, X, ocean_cost_map, cmap=plt.cm.viridis, linewidth=0.2)
		# self.cnt += 1	
		# filename='./imgs/'+str(self.cnt)+'.png'
		# plt.savefig(fname=filename, dpi=96)
		# plt.gca()
		# plt.close(fig)

		return mvm[0], mvm[1]

	def constructDestination(self, Z, X, Y, destination:Position):
		Z = Z - 15/(1 + np.sqrt( (X - destination.x)**2 + (Y - destination.y)**2))
		return Z
	def constructObstacle(self, Z, X, Y, destination:Position):
		Z = Z + 15/(1 + np.sqrt( (X - destination.x)**2 + (Y - destination.y)**2))
		return Z
	def normaDist(self, pointA, pointB):
		return np.sqrt( (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2 )

	def move_to_position(self, external_factors: List[Tuple[AgentTypes, Position]]):

		if (len(external_factors) == 0):
			return

		idx = -1
		for i, factor in enumerate(external_factors):
			if factor[1] == self.position:
				idx = i
				break
		external_factors.pop(idx)
		
		self.external_factors = external_factors
		self.movement_forces = self.compile_forces()
		self.move()

	def move(self):
		if self.position.y + self.movement_forces[1] > HEIGHT + POS_OFFSET:
			self.position.y = -POS_OFFSET
		elif self.position.y + self.movement_forces[1] < -POS_OFFSET:
			self.position.y = HEIGHT + POS_OFFSET
		else:
			self.position.y += self.movement_forces[1]
		
		if self.position.x + self.movement_forces[0] > WIDTH + POS_OFFSET:
			self.position.x = -POS_OFFSET
		elif self.position.x + self.movement_forces[0] < -POS_OFFSET:
			self.position.x = WIDTH + POS_OFFSET
		else:
			self.position.x += self.movement_forces[0]

