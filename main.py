from __future__ import annotations
from cgi import test

from turtle import pos, position
from typing import Dict
from typing import Tuple
from typing import List

import numpy as np

import time

from utils import calculate_distance, reached_destination


import numpy.matlib
import random
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]

from mpi4py import MPI


from agent import Agent

from utils import Payload, AgentTypes, Position


import uuid 

WIDTH, HEIGHT = 1000, 1000
POS_OFFSET = 50
BATCH_SIZE = 3
NUM_EPOCHS = 100


def print_agent_states(agents: Dict[uuid.UUID, Agent]):
	print('AGENT STATES:')
	for agent in agents.values():
		print(agent.id, agent.position)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:

	time_elapsed = []
	agents: Dict[uuid.UUID, Agent] = {}
	epoch = 0

	run = True
	while run:
		try:
			epoch += 1
			if epoch > NUM_EPOCHS:
				run = False
				break

			begin = time.perf_counter()

			req = comm.irecv(source=MPI.ANY_SOURCE)
			response = req.wait()
			rank, data, deleted_agents, epoch = response

			for datum in data:
				if isinstance(datum, Agent):
					agents[datum.id] = datum
				else:
					print('Unknown agent')

			external_factors = []
			for agent in agents.values():
				external_factors.append((agent.type, agent.position))
			
			for slave in range(1, comm.Get_size()):
				req = comm.isend(external_factors, slave)
				req.wait()

			for deleted_agent in deleted_agents:
				if isinstance(deleted_agent, Agent):
					if deleted_agent.id in agents.keys():
						agents.pop(deleted_agent.id)

			print(epoch)

			end = time.perf_counter()

			time_elapsed.append((begin, end))

			
		except BaseException as err:
			print(f"Unexpected {err=}, {type(err)=}")
			raise
	
	print(time_elapsed)
	total_time = 0
	for t in time_elapsed:
		total_time += abs(t[1] - t[0])
	print('Total time: ', total_time)

elif rank % 4 == 0:

	epoch = 1
	rank = comm.Get_rank()
	sharks = [Agent(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)),rank,AgentTypes.SHARK) for _ in range(0,2)]
	deleted_sharks = []

	req = comm.isend((rank, sharks, deleted_sharks, epoch), 0)
	req.wait()

	while True:

		resp = comm.irecv(source=0)
		external_factors = resp.wait()

		for shark in sharks:
			shark.age()
			shark.move_to_position(external_factors)
			if False:
				sharks.remove(shark)
				deleted_sharks.append(shark)
			elif False:
				current_pos = shark.position
				sharks.append(Agent(Position(current_pos.x, current_pos.y),rank,AgentTypes.SHARK))
			
		epoch += 1

		req = comm.isend((rank, sharks, deleted_sharks, epoch), 0)
		req.wait()
			

else:

	epoch = 1
	rank = comm.Get_rank()
	fish = [Agent(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)),rank,AgentTypes.FISH)for _ in range(0,7)]
	deleted_fish = []

	req = comm.isend((rank, fish, deleted_fish, epoch), 0)
	req.wait()
	while True:

		req = comm.irecv(source=0)
		external_factors = req.wait()
		
		if len(fish) == 0:
			fish.append(Agent(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)),rank,AgentTypes.FISH))
			

		for f in fish:
			f.age()
			f.move_to_position(external_factors)
			if False:
				fish.remove(f)
				deleted_fish.append(f)
			elif False:
				current_pos = f.position
				fish.append(Agent(Position(current_pos.x, current_pos.y),rank,AgentTypes.FISH))
			
		epoch += 1

		req = comm.isend((rank, fish, deleted_fish, epoch), 0)
		req.wait()

