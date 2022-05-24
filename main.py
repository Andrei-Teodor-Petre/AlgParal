from __future__ import annotations
from cgi import test

from turtle import pos, position
from typing import Dict
from typing import Tuple
from typing import List

import time

from utils import calculate_distance


import numpy.matlib
import random
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]

from mpi4py import MPI


from agent import Agent, AgentTypes, Position, Payload

import uuid 

WIDTH, HEIGHT = 1000, 1000
POS_OFFSET = 50
BATCH_SIZE = 3





def reached_destination(position: Position, destination: Position, alignment_error: int) -> bool :
	if( abs(position.x - destination.x) < alignment_error  and abs(position.y - destination.y) < alignment_error):
		return True
	return False


def print_agent_states(agents: Dict[uuid.UUID, Agent]):
	print('AGENT STATES:')
	for agent in agents.values():
		print(agent.id, agent.get_position())

comm = MPI.COMM_WORLD
rank = comm.Get_rank()



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

	agents: Dict[uuid.UUID, Agent] = {}
	test_shark = Agent(Position(random.randint(0,WIDTH - 1), random.randint(0,HEIGHT - 1)),0,AgentTypes.SHARK)
	agents[test_shark.id] = test_shark

	run = True
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
			

		except BaseException as err:
			print(f"Unexpected {err=}, {type(err)=}")
			raise

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

