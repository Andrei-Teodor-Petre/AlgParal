from __future__ import annotations
from cgi import test
import enum
import string
from turtle import pos, position
from typing import Dict
from typing import Tuple
from typing import List

import pygame
import time
import math

from scipy import rand
from utils import scale_image, blit_rotate_center

import numpy as np
import numpy.matlib
import random
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]
from scipy.special import comb
from matplotlib import cm
from mpi4py import MPI

from enum import Enum


import uuid 

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

	def reached_destination(self, destination: Position, alignment_error: int) -> bool:
		if( abs(self.position.x - destination.x) < alignment_error  and abs(self.position.y - destination.y) < alignment_error):
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

	def draw(self, win):
		img = None
		if self.type is AgentTypes.SHARK:
			img = SHARK
		elif self.type is AgentTypes.FISH:
			img = FISH
		if img is None:
			raise Exception('Unknown img type for agent', self.id)
		blit_rotate_center(win, img, (self.position.x, self.position.y), 0)


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


	
def reached_destination(position: Position, destination: Position, alignment_error: int) -> bool :
	if( abs(position.x - destination.x) < alignment_error  and abs(position.y - destination.y) < alignment_error):
		return True
	return False

def draw(win, images, agents: List[Agent]):
	for img, pos in images:
		win.blit(img, pos)

	for agent in agents:
		agent.draw(win)
	pygame.display.update()

def print_agent_states(agents: Dict[uuid.UUID, Agent]):
	print('AGENT STATES:')
	for agent in agents.values():
		print(agent.id, agent.get_position())

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

OCEAN = scale_image(pygame.image.load("imgs/ocean.jpg"), 1)
SHARK = scale_image(pygame.image.load("imgs/shark.png"), 0.15)
FISH = scale_image(pygame.image.load("imgs/fish.png"), 0.03)

def calculate_distance(pos1: Position, pos2: Position):
	return math.sqrt( (pos2.x - pos1.x)**2 + (pos2.y - pos1.y) **2 )

def get_nearest_agent(current_agent: Agent, agents: List[Tuple[AgentTypes, Position]]):
	nearest_agent = None
	nearest_agent_dist = None
	for agent in agents:
		if agent[0] != current_agent.type:
			if nearest_agent is None:
				nearest_agent = agent
				nearest_agent_dist = calculate_distance(current_agent.get_position(), agent[1])
			else:
				dist = calculate_distance(current_agent.get_position(), agent[1])
				if dist < nearest_agent_dist:
					nearest_agent = agent
					nearest_agent_dist = dist
	return nearest_agent[1]

if rank == 0:

	WIN = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("Sharks and fish simulation")

	FPS = 60

	run = True
	clock = pygame.time.Clock()
	images = [(OCEAN, (0, 0))]

	agents: Dict[uuid.UUID, Agent] = {}
	test_shark = Agent(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)),0,AgentTypes.SHARK)
	agents[test_shark.id] = test_shark

	while run:
		try:

			req = comm.irecv(source=MPI.ANY_SOURCE)
			response = req.wait()
			rank, data, deleted_agents, epoch = response
			print('Got msg from rank', rank, ':', len(data), ' agents')

			print('Ocean Epoch: ', epoch)
			print('Num agents BEFORE update: ', len(agents))
			for datum in data:
				if isinstance(datum, Agent):
					agents[datum.id] = datum
				else:
					print('Unknown agent')
			print('Num agents AFTER update: ', len(agents))

			for deleted_agent in deleted_agents:
				if isinstance(deleted_agent, Agent):
					if deleted_agent.id in agents.keys():
						agents.pop(deleted_agent.id)

			external_factors = []
			for agent in agents.values():
				external_factors.append((agent.type, agent.get_position()))
			comm.isend(external_factors, rank)

			print_agent_states(agents)
			
			#inp = input('NEXT ITTER')

			draw(WIN, images, agents.values())
			clock.tick(FPS)
			
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					run = False
					break
			
			keys = pygame.key.get_pressed()
			moved = False

			if keys[pygame.K_a] or keys[pygame.K_LEFT]:
				test_shark.accelerateRight()
				moved = True
			elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
				test_shark.accelerateLeft()
				moved = True
			else:
				test_shark.decelerateOX()

			if keys[pygame.K_w] or keys[pygame.K_UP]:
				test_shark.accelerateUp()
				moved = True
			elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
				test_shark.accelerateDown()
				moved = True
			else:
				test_shark.decelerateOY()

			if not moved:
				test_shark.decelerate()

		except BaseException as err:
			print(f"Unexpected {err=}, {type(err)=}")
			raise

	pygame.quit()

elif rank % 9 == 0:

	epoch = 1
	rank = comm.Get_rank()
	sharks = [Agent(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)),rank,AgentTypes.SHARK)]
	deleted_sharks = []

	while True:

		req = comm.isend((rank, sharks, deleted_sharks, epoch), 0)
		req.wait()

		for shark in sharks:
			print('EPOCH ', epoch)
			shark.age()
			shark.move_to_position(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)))
			if shark.should_die():
				sharks.remove(shark)
				deleted_sharks.append(shark)
			elif shark.can_breed():
				print(f'SHARK FROM RANK {rank} CAN BREED')
				current_pos = shark.get_position()
				sharks.append(Agent(Position(current_pos.x + 50, current_pos.y + 50),rank,AgentTypes.SHARK))
			
		epoch += 1
			

else:

	epoch = 1
	rank = comm.Get_rank()
	fish = [Agent(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)),rank,AgentTypes.FISH)]
	deleted_fish = []

	while True:

		req = comm.isend((rank, fish, deleted_fish, epoch), 0)
		req.wait()

		if len(fish) == 0:
			print('NO MORE FISH / REFRESHING')
			fish.append(Agent(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)),rank,AgentTypes.FISH))
			

		for f in fish:
			print('EPOCH ', epoch)
			f.age()
			f.move_to_position(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)))
			if f.should_die():
				fish.remove(f)
				deleted_fish.append(f)
			
		epoch += 1

