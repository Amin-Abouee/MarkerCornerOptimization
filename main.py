#! /usr/local/bin/python

import numpy as np
from marker_corner_optimization import MarkerCornerOptimization
import sys
import logging

if __name__ == "__main__":

	logger = logging.getLogger("_CORNER_OPTIMIZATION_")
	logger.setLevel(logging.DEBUG)

	fh = logging.FileHandler(sys.argv[1])
	fh.setLevel(logging.DEBUG)
	
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	fh.setFormatter(formatter)

	logger.addHandler(ch)
	logger.addHandler(fh)

	marker_optimizer = MarkerCornerOptimization()
	with open(sys.argv[2]) as csvfile:
		for index, line in enumerate(csvfile):
			logger.info("\nCase: " + str(index + 1))
			line = line.rstrip()
			raw_data = line.split(';')
			LENGTH_SIDE = float(raw_data[4])
			x = []
			y = []
			for i in xrange(4):
				x_t, y_t = raw_data[i].strip('[]').split(',')
				x.append(float(x_t))
				y.append(float(y_t))

			marker_optimizer.set_parameters(x, y, LENGTH_SIDE, logger)
			marker_optimizer.optimize()
			marker_optimizer.print_result()
			marker_optimizer.draw_graph("%sout_%04d.jpg" % (sys.argv[3],index))

	# x = [599.123, 628.801, 689.092, 658.951]
	# y = [446.382, 386.384, 416.144, 476.233]
	# x = [0, 1, 1, 0]
	# y = [0, 0, 1, 1]
	# LENGTH_SIDE = 66.931
	# LENGTH_SIDE = 1.0
	# marker_optimizer.set_parameters(x, y, LENGTH_SIDE, logger)
	# marker_optimizer.optimize()
	# marker_optimizer.print_result()
	# marker_optimizer.draw_graph("out.jpg")
