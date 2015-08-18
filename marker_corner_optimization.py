import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import logging

class MarkerCornerOptimization:
	def __init__ (self, x, y, length_side, logger = None):
		self.x = x
		self.y = y
		self.LENGTH_SIDE = length_side
		self.logger = logger

	def __init__ (self):
		pass

	def set_parameters (self, x, y, length_side, logger = None):
		self.x = x
		self.y = y
		self.LENGTH_SIDE = length_side
		self.logger = logger

	def cost_fuction (self, c):
		m0 = ((c[1] - c[0])*(c[2] - c[1]) + (c[5] - c[4])*(c[6] - c[5])) ** 2
		m1 = ((c[2] - c[1])*(c[3] - c[2]) + (c[6] - c[5])*(c[7] - c[6])) ** 2
		m2 = ((c[3] - c[2])*(c[0] - c[3]) + (c[7] - c[6])*(c[4] - c[7])) ** 2
		m3 = ((c[0] - c[3])*(c[1] - c[0]) + (c[4] - c[7])*(c[5] - c[4])) ** 2

		m4 = (math.sqrt((c[1] - c[0]) ** 2 + (c[5] - c[4]) ** 2) - self.LENGTH_SIDE) ** 2
		m5 = (math.sqrt((c[2] - c[1]) ** 2 + (c[6] - c[5]) ** 2) - self.LENGTH_SIDE) ** 2
		m6 = (math.sqrt((c[3] - c[2]) ** 2 + (c[7] - c[6]) ** 2) - self.LENGTH_SIDE) ** 2
		m7 = (math.sqrt((c[0] - c[3]) ** 2 + (c[4] - c[7]) ** 2) - self.LENGTH_SIDE) ** 2

		return m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7

	def optimize (self):
		c = self.x + self.y
		res = minimize(self.cost_fuction, c, method='powell', options={'xtol': 1e-8, 'disp': True})
		self.opt_x = res.x[:4]
		self.opt_y = res.x[4:]

	def print_result (self):
		self.logger.info("INIT X: " + str(self.x))
		self.logger.info("INIT Y: " + str(self.y))
		self.logger.info("Optimized X: " + str(self.opt_x[:4]))
		self.logger.info("Optimized Y: " + str(self.opt_y[:4]))

	def draw_graph (self, file_name):
		fig = plt.figure(figsize = (5,9), dpi = 300)
		ax = fig.add_subplot(111)
		ax.plot(self.x, self.y, "ro")
		ax.plot(self.opt_x, self.opt_y, "bo")
		plt.axis("equal")
		plt.savefig(file_name)
		self.logger.info("Image saved in" + file_name)
