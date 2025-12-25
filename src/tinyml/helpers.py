import time
import os


def _timeit(func):
	def enc_func(*args):
		I = int(os.environ.get("I", 10))
		if I == -1:
			while 1:
				t = timeit(func, *args)
				print(func.__name__, t)
		else:		
			for _ in range(I):
				t = timeit(func, *args)
				print(t)

	return enc_func
		

def timeit(func, *args):
	st = time.monotonic()
	func(*args)
	et = time.monotonic()
	
	return et - st # in seconds


def generate(lower=1, upper=100, size = (1, 1)):
	import numpy as np
	matrix = np.random.randint(lower, upper, size)
	
	return matrix
	
	
