from enum import Enum
from typing import List, Tuple
import uuid
import math
from mpi4py import MPI

import numpy as np

from utils import calculate_distance, reached_destination
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#we should NOT need these here


WIDTH, HEIGHT = 1000, 1000
POS_OFFSET = 50
BATCH_SIZE = 3

class AgentTypes(Enum):
	OCEAN = 0
	SHARK = 1
	FISH  = 2

class Position:
	def __init__(self,x,y):
		self.x = x
		self.y = y
	def __str__(self):
		return f'[X:{self.x}, Y:{self.y}]'

class Payload:
	def __init__(self, id: uuid.UUID, type: AgentTypes):
		self.id = id
		self.type = type

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

	def age(self):
		self.current_age += 1

	def can_breed(self) -> bool:
		if self.type is AgentTypes.SHARK:
			return False
		elif self.type is AgentTypes.FISH:
			return self.current_age >= 10 #and random.randint(0, 9) == 7

	def should_die(self) -> bool:
		if self.type is AgentTypes.SHARK:
			if self.fish_collision():
				self.current_age -= 10
			return self.current_age > 100
		elif self.type is AgentTypes.FISH:
			return self.shark_collision()

	def fish_collision(self) -> bool:
		if self.external_factors is not None:
			for factor in self.external_factors:
				agent_type, position = factor
				if agent_type is AgentTypes.FISH and calculate_distance(position, self.position) <= 50:
					return True
		return False

	def shark_collision(self) -> bool:
		if self.external_factors is not None:
			for factor in self.external_factors:
				agent_type, position = factor
				if agent_type is AgentTypes.SHARK and calculate_distance(position, self.position) <= 50:
					return True
		return False

	def get_payload(self) -> Payload:
		# data passed around through comm
		# should return everything that node 0 (i.e the OCEAN) needs to display a simulation itteration
		return Payload(self.id, self.type)

	def get_position(self):
		return self.position
	
	def request_external_factors(self):
		req = comm.irecv(source=0)
		external_factors = req.wait()
		print('Got external factors', self.id)
		return external_factors

	def compile_forces(self):
		self.external_factors = self.request_external_factors()
		external_force = self.compile_external_force()
		#gravity = self.compile_gravity(external_factors)
		gravity = [0, 0]
		return (external_force[0] + gravity[0]), (external_force[1] + gravity[1])

	def compile_external_force(self):
		return 1, 1

	def compile_gravity(self, external_factors: List[Tuple[AgentTypes, Position]]):
		
		x = np.arange(0,WIDTH,1)
		y = np.arange(0,HEIGHT,1)
		X,Y = np.meshgrid(x,y)

		ocean_cost_map = np.zeros(len(X))

		fish = []
		sharks = []
		for item in external_factors:
			if item[0] is AgentTypes.SHARK:
				sharks.append(item[1])
			else:
				fish.append(item[1])

		if self.type is AgentTypes.SHARK:
			for fish_dest in fish:
				ocean_cost_map = self.constructDestination(ocean_cost_map, X, Y, fish_dest)
		elif self.type is AgentTypes.FISH:
			for shark_obstacle in sharks:
				ocean_cost_map = self.constructObstacle(ocean_cost_map, X, Y, shark_obstacle)
		
		dx,dy = np.gradient(ocean_cost_map,1,1)

		pointI = Position(x[int(self.position.x)]-1, y[int(self.position.y)]-1)

		u = -dx[int(pointI.x),int(pointI.y)]
		v = -dy[int(pointI.x),int(pointI.y)]

		pointI.x = pointI.x + u
		pointI.y = pointI.y + v

		destination = get_nearest_agent(self, external_factors)
		atractiveComp = ([pointI.x, pointI.y] - np.array([destination.x, destination.y])) / calculate_distance(pointI, destination)

		pointI = np.array([self.position.x, self.position.y])
		repulsiveComp = [0,0]
		obstacle = np.array([[p.x, p.y] for p in sharks])
		for i in range(len(obstacle > 0)):
			repulsiveComp += (pointI - obstacle[i]) / self.normaDist(pointA=pointI, pointB= obstacle[i])

		mvm = -(atractiveComp + repulsiveComp)

		print("Attractive: ", atractiveComp)
		print("Repulsive: ", repulsiveComp)
		print("Mvm: ", mvm)

		return mvm[0], mvm[1]

	def constructDestination(self, Z, X, Y, destination:Position):
		Z = Z - 15/(1 + np.sqrt( (X - destination.x)**2 + (Y - destination.y)**2))
		return Z
	def constructObstacle(self, Z, X, Y, destination:Position):
		Z = Z + 15/(1 + np.sqrt( (X - destination.x)**2 + (Y - destination.y)**2))
		return Z
	def normaDist(self, pointA, pointB):
		return np.sqrt( (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2 )

	def move_to_position(self, destination: Position):
		external_force = self.compile_forces()
		self.destination = destination
		if( not reached_destination(self.position, self.destination, self.alignment_error) ):
			if(self.position.x - destination.x < 0):
				self.accelerateLeft(external_force)
			else:
				self.accelerateRight(external_force)

			if(self.position.y - destination.y < 0):
				self.accelerateDown(external_force)
			else:
				self.accelerateUp(external_force)
		else:
			self.decelerate(external_force)


	def move(self):
		if self.position.y + self.velocity_Y > HEIGHT + POS_OFFSET:
			self.position.y = -POS_OFFSET
		elif self.position.y + self.velocity_Y < -POS_OFFSET:
			self.position.y = HEIGHT + POS_OFFSET
		else:
			self.position.y += self.velocity_Y
		
		if self.position.x + self.velocity_X > WIDTH + POS_OFFSET:
			self.position.x = -POS_OFFSET
		elif self.position.x + self.velocity_X < -POS_OFFSET:
			self.position.x = WIDTH + POS_OFFSET
		else:
			self.position.x += self.velocity_X

	def accelerateUp(self, external_force = None):
		if external_force is not None:
			velocity = -self.max_velocity + external_force[1]
		else:
			velocity = -self.max_velocity
		if(self.velocity_Y > velocity):
			self.velocity_Y -= self.acceleration
		self.move()

	def accelerateDown(self, external_force = None):
		if external_force is not None:
			velocity = self.max_velocity + external_force[1]
		else:
			velocity = self.max_velocity
		if(self.velocity_Y < velocity):
			self.velocity_Y += self.acceleration
		self.move()

	def accelerateLeft(self, external_force = None):
		if external_force is not None:
			velocity = self.max_velocity + external_force[0]
		else:
			velocity = self.max_velocity
		if(self.velocity_X < velocity):
			self.velocity_X += self.acceleration
		self.move()

	def accelerateRight(self, external_force = None):
		if external_force is not None:
			velocity = -self.max_velocity + external_force[0]
		else:
			velocity = -self.max_velocity
		if(self.velocity_X > velocity):
			self.velocity_X -= self.acceleration
		self.move()

	def decelerateOX(self, external_force = None):

		if external_force is not None:
			control_value = external_force[0]
		else:
			control_value = 0

		if(abs(self.velocity_X) < self.acceleration): self.velocity_X = control_value
		if(self.velocity_X == control_value): return

		if(self.velocity_X > control_value):
			self.velocity_X = max(self.velocity_X - self.acceleration, control_value)
		else:
			self.velocity_X = min(self.velocity_X + self.acceleration, control_value)

	def decelerateOY(self, external_force = None):

		if external_force is not None:
			control_value = external_force[1]
		else:
			control_value = 0

		if abs(self.velocity_Y) < self.acceleration: self.velocity_Y = control_value
		if self.velocity_Y == control_value: return

		if self.velocity_Y > control_value:
			self.velocity_Y = max(self.velocity_Y - self.acceleration, control_value)
		else:
			self.velocity_Y = min(self.velocity_Y + self.acceleration, control_value)
			
		if external_force is not None:
			self.velocity_Y -= external_force[1]
			
	def decelerate(self, external_force = None):
		self.decelerateOX(external_force)
		self.decelerateOY(external_force)
		self.move()


	
