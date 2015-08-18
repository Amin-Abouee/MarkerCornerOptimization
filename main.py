#! /usr/local/bin/python

import numpy as np
from marker_corner_optimization import MarkerCornerOptimization
import sys

if __name__ == "__main__":

	marker_optimizer = MarkerCornerOptimization()
	with open(sys.argv[1]) as csvfile:
		for index, line in enumerate(csvfile):
			line = line.rstrip()
			raw_data = line.split(';')
			LENGTH_SIDE = float(raw_data[4])
			x = []
			y = []
			for i in xrange(4):
				x_t, y_t = raw_data[i].strip('[]').split(',')
				x.append(float(x_t))
				y.append(float(y_t))

			marker_optimizer.set_parameters(x, y, LENGTH_SIDE)
			marker_optimizer.optimize()
			marker_optimizer.print_result()
			marker_optimizer.draw_graph("%sout_%3d.jpg" % (sys.argv[2],index))