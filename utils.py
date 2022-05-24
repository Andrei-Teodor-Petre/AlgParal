import math
from agent import Position


def calculate_distance(pos1: Position, pos2: Position):
	return math.sqrt( (pos2.x - pos1.x)**2 + (pos2.y - pos1.y) **2 )


def reached_destination(self, destination: Position, alignment_error: int) -> bool:
	if( abs(self.position.x - destination.x) < alignment_error  and abs(self.position.y - destination.y) < alignment_error):
		return True
	return False
