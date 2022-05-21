from __future__ import annotations
import enum
import string
from turtle import pos

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


class Type(Enum):
	OCEAN = 0
	SHARK = 1
	FISH  = 2

class Position:
	def __init__(self,x,y):
		self.x = x
		self.y = y




class Agent:
		
	def __init__(self, posi: Position, _rank: int, image: pygame.Surface, type: Type = None):
		
		self.Id = uuid.uuid4()

		self.rank = _rank
		self.img = image

		if image == SHARK:
			self.type = Type.SHARK
		else:
			self.type = Type.FISH

		self.position = posi
		self.destination = None

		self.max_velocity = 4
		self.acceleration = 0.1
		self.velocity_X = 0
		self.velocity_Y = 0
		self.alignment_error = 40

		#this can be counted in while cycles
		self.age = 0



	def get_position(self):
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


#nush daca avem nevoie de astea dar le stergem la sf daca nu le folosim
class Ocean(Agent):
	pass
class Shark(Agent):
	pass
class Fish(Agent):
	pass
	
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

OCEAN = scale_image(pygame.image.load("imgs/ocean.jpg"), 1.5)

SHARK = scale_image(pygame.image.load("imgs/shark.png"), 0.1)
FISH = scale_image(pygame.image.load("imgs/fish.png"), 0.03)

if rank == 0:



	WIDTH, HEIGHT = 1600, 1000
	WIN = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("Sharks and fish simulation")

	FPS = 60

	run = True
	clock = pygame.time.Clock()
	images = [(OCEAN, (0, 0))]

	Agents = []

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

			#broadcast alive and dead status

			#listen for shit

			#process

			#send shit back


		except BaseException as err:
			print(f"Unexpected {err=}, {type(err)=}")
			raise

	pygame.quit()

elif rank % 23 == 0:
	# shark 
	# se instantiaza un agent -> shark cu pozitii random pe ocean (cu conditia sa fie libere i.e apa nu un agent existent)
	randx = random.randint(0,1599)
	randy = random.randint(0,999)
	rank = comm.Get_rank()
	shark = Agent(Position(randx, randy),rank,SHARK)	
	
	# comunica cu ocean

	
	# while liveCond
	while True:
		#get alive status -> if dead maybe spawn again idk, ca sa nu irosim procesul ca nush how the fuck mai instantiem altele dupa comanda initiala din terminal


		#get nearest shark


		#update internal position away from it


		#comm the position to the ocean

		pass
		# are un set de reguli dupa care traieste
		# un movement function + toate compunerile cu exteriorul (gravitatie, nearest neighbors etc.)

else:
	#fish
	# se instantiaza un agent -> fish cu pozitii random pe ocean (cu conditia sa fie libere i.e apa nu un agent existent)
	randx = random.randint(0,1599)
	randy = random.randint(0,999)
	rank = comm.Get_rank()
	shark = Agent(Position(randx, randy),rank,FISH)	
	# comunica cu ocean
	comm.Isend(shark,0)



	# while liveCond
	while True:
		#get alive status -> if dead maybe spawn again idk, ca sa nu irosim procesul ca nush how the fuck mai instantiem altele dupa comanda initiala din terminal


		#get nearest shark


		#update internal position away from it


		#comm the position to the ocean

		pass
		# are un set de reguli dupa care traieste
		# un movement function + toate compunerile cu exteriorul (gravitatie, nearest neighbors etc.)

