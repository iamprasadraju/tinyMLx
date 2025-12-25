import time
import os
import tracemalloc
from functools import wraps

def timeit(func):
	def enc_func(*args):
		I = int(os.environ.get("I", 10))
		if I == -1:
			while 1:
				t = _timeit(func, *args)
				print(func.__name__, t)
		else:		
			for _ in range(I):
				t = _timeit(func, *args)
				print(t)

	return enc_func
		

def _timeit(func, *args):
	st = time.monotonic()
	func(*args)
	et = time.monotonic()
	
	return et - st # in seconds


def generate(lower=1, upper=100, size = (1, 1)):
	import numpy as np
	matrix = np.random.randint(lower, upper, size)
	
	return matrix
	
	
def memprofile(func):
	@wraps(func)
	def wrapper(*args):
		tracemalloc.start()
		_ = func(*args)
		
		snapshot = tracemalloc.take_snapshot()
		top_stats = snapshot.statistics('lineno')
		
		for stat in top_stats[:10]:
		    print(stat)
		tracemalloc.stop()
		
		return _
	return wrapper