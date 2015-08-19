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

		d1_x = self.opt_x[2] - self.opt_x[0]
		d1_y = self.opt_y[2] - self.opt_y[0]
		# d2_x = self.opt_x[3] - self.opt_x[1]
		# d2_y = self.opt_y[3] - self.opt_y[1]
		# c1 = d1_y * self.opt_x[0] - d1_x * self.opt_y[0]
		# c2 = d2_y * self.opt_x[1] - d2_x * self.opt_y[1]

		# if (d2_y * d1_x is not d1_y * d2_x):
			# a_mat = np.array([[d1_y, -d1_x], [d2_y, -d2_x]])
			# b_mat = np.array([c1, c2])
			# x, y = np.linalg.solve(a_mat, b_mat)
			# self.opt_center = np.array([x,y])

		self.opt_center = np.array([self.opt_x[0] + d1_x/2, self.opt_y[0] + d1_y/2])

	def print_result (self):
		self.logger.info("Init X: " + str(self.x))
		self.logger.info("Init Y: " + str(self.y))
		self.logger.info("Optimized X: " + str(self.opt_x))
		self.logger.info("Optimized Y: " + str(self.opt_y))
		self.logger.info("Center: " + str(self.opt_center))
		# self.logger.info("Center2: " + str(self.t_center_x) + "  " + str(self.t_center_y))

		self.logger.info("D1: " + str(math.sqrt((self.opt_x[1] - self.opt_x[0]) ** 2 + (self.opt_y[1] - self.opt_y[0]) ** 2)))
		self.logger.info("D2: " + str(math.sqrt((self.opt_x[2] - self.opt_x[1]) ** 2 + (self.opt_y[2] - self.opt_y[1]) ** 2)))
		self.logger.info("D3: " + str(math.sqrt((self.opt_x[3] - self.opt_x[2]) ** 2 + (self.opt_y[3] - self.opt_y[2]) ** 2)))
		self.logger.info("D4: " + str(math.sqrt((self.opt_x[0] - self.opt_x[3]) ** 2 + (self.opt_y[0] - self.opt_y[3]) ** 2)))

		self.logger.info("A1: " + str(math.degrees(math.acos((self.opt_x[1] - self.opt_x[0])*(self.opt_x[2] - self.opt_x[1]) + (self.opt_y[1] - self.opt_y[0])*(self.opt_y[2] - self.opt_y[1])))))
		self.logger.info("A1: " + str(math.degrees(math.acos((self.opt_x[2] - self.opt_x[1])*(self.opt_x[3] - self.opt_x[2]) + (self.opt_y[2] - self.opt_y[1])*(self.opt_y[3] - self.opt_y[2])))))
		self.logger.info("A1: " + str(math.degrees(math.acos((self.opt_x[3] - self.opt_x[2])*(self.opt_x[0] - self.opt_x[3]) + (self.opt_y[3] - self.opt_y[2])*(self.opt_y[0] - self.opt_y[3])))))
		self.logger.info("A1: " + str(math.degrees(math.acos((self.opt_x[0] - self.opt_x[3])*(self.opt_x[1] - self.opt_x[0]) + (self.opt_y[0] - self.opt_y[3])*(self.opt_y[1] - self.opt_y[0])))))

	def draw_graph (self, file_name):
		fig = plt.figure(figsize = (5,9), dpi = 300)
		ax = fig.add_subplot(111)
		ax.plot(self.x, self.y, "ro")
		ax.plot(self.opt_x, self.opt_y, "bo")
		ax.plot(self.opt_center[0], self.opt_center[1], "go")
		plt.axis("equal")
		plt.savefig(file_name)
		self.logger.info("Image saved in" + file_name)
