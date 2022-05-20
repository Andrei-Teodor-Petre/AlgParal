from __future__ import annotations
import string
from turtle import pos

import pygame
import time
import math
from utils import scale_image, blit_rotate_center

import numpy as np
import numpy.matlib
import random
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]
from scipy.special import comb
from matplotlib import cm
from mpi4py import MPI

import uuid 

class Position:
	def __init__(self,x,y):
		self.x = x
		self.y = y

class Ocean:
	pass

class Agent:
		
	def __init__(self, posi: Position, ocean: Ocean, image: pygame.Surface, type: str = None):
		
		self.Id = uuid.uuid4()
		self.position = posi
		
		self.img = image
		self.destination = None

		self.max_velocity = 4
		self.acceleration = 0.1
		self.velocity_X = 0
		self.velocity_Y = 0
		self.alignment_error = 40


	def get_position(self):
		#we need this so other drones can reference it
		return self.position
	
	def move_to_position(self, destination: Position):
		self.destination = destination
		if( not reached_destination(self.position, self.destination, self.alignment_error) ):
			if(self.position.x - destination.x < 0):
				self.accelerateLeft()
			else:
				self.accelerateRight()

			if(self.position.y - destination.y < 0):
				self.accelerateDown()
			else:
				self.accelerateUp()
		else:
			self.decelerate()

		
	def draw(self, win):
		blit_rotate_center(win, self.img, (self.position.x, self.position.y), 0)


	def move(self):
		self.position.y += self.velocity_Y
		self.position.x += self.velocity_X

	def accelerateUp(self):
		if(self.velocity_Y > -self.max_velocity):
			self.velocity_Y -= self.acceleration
		self.move()

	def accelerateDown(self):
		if(self.velocity_Y < self.max_velocity):
			self.velocity_Y += self.acceleration
		self.move()

	def accelerateLeft(self):
		if(self.velocity_X < self.max_velocity):
			self.velocity_X += self.acceleration
		self.move()

	def accelerateRight(self):
		if(self.velocity_X > -self.max_velocity):
			self.velocity_X -= self.acceleration
		self.move()

	def decelerateOX(self):
		if(abs(self.velocity_X) < self.acceleration): self.velocity_X = 0
		if(self.velocity_X == 0): return

		if(self.velocity_X > 0):
			self.velocity_X = max(self.velocity_X - self.acceleration, 0)
		else:
			self.velocity_X = min(self.velocity_X + self.acceleration, 0)

	def decelerateOY(self):
		if abs(self.velocity_Y) < self.acceleration: self.velocity_Y = 0
		if self.velocity_Y == 0: return

		if self.velocity_Y > 0:
			self.velocity_Y = max(self.velocity_Y - self.acceleration, 0)
		else:
			self.velocity_Y = min(self.velocity_Y + self.acceleration, 0)
			

	def decelerate(self):

		self.decelerateOX()
		self.decelerateOY()
		self.move()
			

	
def reached_destination(position: Position, destination: Position, alignment_error: int) -> bool :
	if( abs(position.x - destination.x) < alignment_error  and abs(position.y - destination.y) < alignment_error):
		return True
	return False

def draw(win, images, Drones):
	for img, pos in images:
		win.blit(img, pos)

	for drn in Drones:
		drn.draw(win)
	pygame.display.update()




comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:

	OCEAN = scale_image(pygame.image.load("imgs/ocean.jpg"), 2.5)

	SHARK = scale_image(pygame.image.load("imgs/shark.png"), 0.1)
	FISH = scale_image(pygame.image.load("imgs/fish.png"), 0.03)

	WIDTH, HEIGHT = 1000, 1000
	WIN = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("Drone simulation")

	FPS = 60

	run = True
	clock = pygame.time.Clock()
	images = [(OCEAN, (0, 0))]

	shark = Agent(Position(180, 200), Ocean(), SHARK,0)
	fish1 = Agent(Position(150, 250), Ocean(), FISH,1)
	fish2 = Agent(Position(250, 250), Ocean(), FISH,7)
	fish3 = Agent(Position(350, 250), Ocean(), FISH,3)

	Agents = [shark, fish1, fish2, fish3]

	while run:
		try:
			
			clock.tick(FPS)

			draw(WIN, images, Agents)

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					run = False
					break

			keys = pygame.key.get_pressed()
			moved = False

			if keys[pygame.K_a] or keys[pygame.K_LEFT] or keys[pygame.K_d] or keys[pygame.K_RIGHT]:
				if keys[pygame.K_a] or keys[pygame.K_LEFT]:
					shark.accelerateRight()
					moved = True
				if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
					shark.accelerateLeft()
					moved = True
			else:
				shark.decelerateOX()

			if keys[pygame.K_w] or keys[pygame.K_UP] or keys[pygame.K_s] or keys[pygame.K_DOWN]:	
				if keys[pygame.K_w] or keys[pygame.K_UP]:
					shark.accelerateUp()
					moved = True
				if keys[pygame.K_s] or keys[pygame.K_DOWN]:
					shark.accelerateDown()
					moved = True
			else:
				shark.decelerateOY()

			# for drone_index in range(1,len(Agents)):
			# 	if(Agents[drone_index].aligned):
			# 		Agents[drone_index].align_to_leader_speed()
			# 	else:
			# 		Agents[drone_index].align_to_leader()



			if not moved:
				shark.decelerate()


		except BaseException as err:
			print(f"Unexpected {err=}, {type(err)=}")
			raise

	pygame.quit()

elif rank % 10 == 0:
	# shark 
	# se instantiaza un agent cu pozitii random pe ocean (cu conditia sa fie libere i.e apa nu un agent existent)
	# comunica cu ocean
	# while liveCond
		# are un set de reguli dupa care traieste
		# un movement function + toate compunerile cu exteriorul (gravitatie, nearest neighbors etc.)

	pass

else: 
	# se instantiaza un agent cu pozitii random pe ocean (cu conditia sa fie libere i.e apa nu un agent existent)
	# comunica cu ocean
	# while liveCond
		# are un set de reguli dupa care traieste
		# un movement function + toate compunerile cu exteriorul (gravitatie, nearest neighbors etc.)
	pass

