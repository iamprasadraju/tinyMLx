import time



def _timeit(func):
	def enc_func(*args):
		t = timeit(func, *args)
		print(t)
	return enc_func
		
		


def timeit(func, *args):
	st = time.monotonic()
	func(*args)
	et = time.monotonic()
	
	return et - st # in seconds


