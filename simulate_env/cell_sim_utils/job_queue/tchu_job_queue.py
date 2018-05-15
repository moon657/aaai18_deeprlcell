import numpy as np
import random

class JobQueue:
	def __init__(self, job_num=10, dT=60, T=500, init_bytes=None):
		self.remain_bytes = np.zeros(job_num)
		self.remian_times = np.ones(job_num) * T
		self.job_num = job_num
		self.dT = dT
		if init_bytes is None:
			self.remain_bytes = self.sample_init_bytes()
		else:
			self.remain_bytes = init_bytes


	def sample_init_bytes(self):
		init_bytes = np.zeros(self.job_num)
		job_means = [1e7, 1e8, 2e8, ]
		job_stds = [] 
		for i in xrange(self.job_num):
			if i < self.job_num/3:
				init_bytes[i] = int(random.lognormvariate(,job_std))