import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math

class MarkerCornerOptimization:
	def __init__ (self, x, y, length_side):
		self.x = x
		self.y = y
		self.LENGTH_SIDE = length_side

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
		print "INIT X: ", self.x
		print "INIT Y: ", self.y
		print "Optimized X: ", self.opt_x[:4]
		print "Optimized Y: ", self.opt_y[:4]

	def draw_graph (self, file_name):
		fig = plt.figure(figsize = (5,9), dpi = 300)
		ax = fig.add_subplot(111)
		ax.plot(self.x, self.y, "ro")
		ax.plot(self.opt_x, self.opt_y, "bo")
		plt.axis("equal")
		plt.savefig(file_name)


if __name__ == "__main__":
	init_x = [599.123, 628.801, 689.092, 658.951]
	init_y = [446.382, 386.384, 416.144, 476.233]
	LENGTH_SIDE = 66.931
	marker_optimizer = MarkerCornerOptimization(init_x, init_y, LENGTH_SIDE)
	marker_optimizer.optimize()
	marker_optimizer.print_result()
	marker_optimizer.draw_graph("out.jpg")



# fig = plt.figure(figsize = (5,9), dpi = 300)
# ax = fig.add_subplot(111)
# ax.plot(init_x, init_y, "ro")

# init_delta_x = [init_x[1]-init_x[0], init_x[2]-init_x[1], init_x[3]-init_x[2], init_x[0]-init_x[3]]
# init_delta_y = [init_y[1]-init_y[0], init_y[2]-init_y[1], init_y[3]-init_y[2], init_y[0]-init_y[3]]
# # print "INIT X: ", init_delta_x
# # print "INIT Y: ", init_delta_y

# angles = [math.atan2(dy,dx) for dy,dx in zip(init_delta_y, init_delta_x)]
# # print [math.degrees(angle) for angle in angles]

# # x = init_x
# # y = init_y
# # for i, angle in enumurate(angles):


# delta_x = [math.cos(theta) * LENGTH_SIDE for theta in angles]
# delta_y = [math.sin(theta) * LENGTH_SIDE for theta in angles]
# # print "X: ", x
# # print "Y: ", y

# # delta_x = [x[1]-x[0], x[2]-x[1], x[3]-x[2], x[0]-x[3]]
# # delta_y = [y[1]-y[0], y[2]-y[1], y[3]-y[2], y[0]-y[3]]
# # print "DELTA X: ", delta_x
# # print "DELTA Y: ", delta_y 

# # print "DELTA X: ", delta_x
# # print "DELTA Y:", delta_y

# # c_x = numpy.array([x[0] - delta_x[0], delta_x[0] - delta_x[1], delta_x[1] - delta_x[2], delta_x[3] - delta_x[2], delta_x[3]])
# # c_y = numpy.array([y[0] - delta_y[0], delta_y[0] - delta_y[1], delta_y[1] - delta_y[2], delta_y[3] - delta_y[2], delta_y[3]])
# # print c

# # A = numpy.array([[3, -1, 0, 0, -1], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [-1, 0 , 0, -1, 2]])
# # x_opt = np.linalg.inv(A).dot(c_x)
# # y_opt = np.linalg.inv(A).dot(c_y)
# # print "OPTIMIZED X:", x_opt
# # print "OPTIMIZED Y:", y_opt

# # ax.plot(x_opt[:-1], y_opt[:-1], "go")
# c = init_x + init_y
# print "INIT: ", c

# def opt(c):
# 	# print "C: ", c
# 	m0 = ((c[1] - c[0])*(c[2] - c[1]) + (c[5] - c[4])*(c[6] - c[5])) ** 2
# 	m1 = ((c[2] - c[1])*(c[3] - c[2]) + (c[6] - c[5])*(c[7] - c[6])) ** 2
# 	m2 = ((c[3] - c[2])*(c[0] - c[3]) + (c[7] - c[6])*(c[4] - c[7])) ** 2
# 	m3 = ((c[0] - c[3])*(c[1] - c[0]) + (c[4] - c[7])*(c[5] - c[4])) ** 2

# 	m4 = (math.sqrt((c[1] - c[0]) ** 2 + (c[5] - c[4]) ** 2) - LENGTH_SIDE) ** 2
# 	m5 = (math.sqrt((c[2] - c[1]) ** 2 + (c[6] - c[5]) ** 2) - LENGTH_SIDE) ** 2
# 	m6 = (math.sqrt((c[3] - c[2]) ** 2 + (c[7] - c[6]) ** 2) - LENGTH_SIDE) ** 2
# 	m7 = (math.sqrt((c[0] - c[3]) ** 2 + (c[4] - c[7]) ** 2) - LENGTH_SIDE) ** 2

# 	return m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7

# # def opt(c):
# # 	# print "C: ", c
# # 	m0 = ((c[0] - init_x[0])*(c[1] - c[0]) + (c[3] - init_y[0])*(c[4] - c[5])) ** 2
# # 	m1 = ((c[1] - c[0])*(c[2] - c[1]) + (c[4] - c[3])*(c[5] - c[4])) ** 2
# # 	m2 = ((c[2] - c[1])*(init_x[0] - c[2]) + (c[5] - c[4])*(init_y[0] - c[5])) ** 2
# # 	m3 = ((init_x[0] - c[2])*(c[0] - init_x[0]) + (init_y[0] - c[5])*(c[3] - init_y[0])) ** 2

# # 	m4 = (math.sqrt((c[0] - init_x[0]) ** 2 + (c[3] - init_y[0]) ** 2) - LENGTH_SIDE) ** 2
# # 	m5 = (math.sqrt((c[1] - c[0]) ** 2 + (c[4] - c[3]) ** 2) - LENGTH_SIDE) ** 2
# # 	m6 = (math.sqrt((c[2] - c[1]) ** 2 + (c[5] - c[4]) ** 2) - LENGTH_SIDE) ** 2
# # 	m7 = (math.sqrt((init_x[0] - c[2]) ** 2 + (init_y[0] - c[5]) ** 2) - LENGTH_SIDE) ** 2

# # 	return m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7

# # x0 = x
# # x0.append(x[0])
# res = minimize(opt, c, method='powell', options={'xtol': 1e-8, 'disp': True})
# print "OPTIMIZED: ", res.x
# # print res.x[:4]
# # print res.x[4:]
# ax.plot(res.x[:4], res.x[4:], "bo")
# # x_min = res.x
# # x_min = [p + init_x[0] for p in x_min]
# # print "SECOND X:", x_min

# # y0 = y
# # y0.append(y[0])
# # res = minimize(opt, y0, delta_y, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
# # y_min = res.x
# # y_min = [p + init_y[0] for p in y_min]
# # print "SECOND Y:", y_min

# # ax.plot(x_min, y_min, "ro")
# plt.axis("equal")
# plt.savefig("all.jpg")

