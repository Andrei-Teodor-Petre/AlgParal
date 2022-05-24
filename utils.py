import math
from enum import Enum
import uuid
class Position:
	def __init__(self,x,y):
		self.x = x
		self.y = y
	def __str__(self):
		return f'[X:{self.x}, Y:{self.y}]'

def calculate_distance(pos1: Position, pos2: Position):
	return math.sqrt( (pos2.x - pos1.x)**2 + (pos2.y - pos1.y) **2 )

def reached_destination(self, destination: Position, alignment_error: int) -> bool:
	if( abs(self.position.x - destination.x) < alignment_error  and abs(self.position.y - destination.y) < alignment_error):
		return True
	return False

class AgentTypes(Enum):
	OCEAN = 0
	SHARK = 1
	FISH  = 2

class Payload:
	def __init__(self, id: uuid.UUID, type: AgentTypes):
		self.id = id
		self.type = type
