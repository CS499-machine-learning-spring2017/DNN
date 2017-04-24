'''
Purpose: Save user input, runtime, and error and print graphs depicting how error changes with user input and runtime changes with data size.
'''
import csv
import numpy as np
import matplotlib.pyplot as plt

num_layers_arr = []
num_subconnected_arr = []
num_nodes_arr = []
num_subconnected_nodes_arr = []
batch_size_arr = []
learning_rate_arr = []
window_size_arr = []
iterations_arr = []
runtime_arr = []
data_size_arr = []
error_arr = []

'''
Purpose: Appends user data to csv file. If file does not exist, create one. 
'''
def save_data(num_layers, num_subconnected,nodes_per_layer, nodes_per_subconnected, batch_size, learning_rate,window_size,iterations, error, runtime, data_size):
	try: 
		with open('SavedData', 'a') as csvfile:
			arr = ([num_layers, num_subconnected, nodes_per_layer, nodes_per_subconnected, batch_size, learning_rate, window_size, iterations, error, runtime, data_size])
			netWriter = csv.writer(csvfile, delimiter=',')
			netWriter.writerow(arr)
	except: 
		with open('SavedData', 'w') as csvfile:
			arr = ([num_layers, num_subconnected, nodes_per_layer, nodes_per_subconnected, batch_size, learning_rate, window_size, iterations, error, runtime, data_size])
			netWriter = csv.writer(csvfile, delimiter=',')
			netWriter.writerow(arr)		

'''
Purpose: Plot various graphs
'''
def graphResults(x, y, plot_name, x_label, y_label):
	font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
	
	plt.plot(x, y, 'k')
	plt.title(plot_name, fontdict=font)
	plt.xlabel(x_label , fontdict=font)
	plt.ylabel(y_label, fontdict=font)

	# Tweak spacing to prevent clipping of ylabel
	plt.subplots_adjust(left=0.15)
	plt.show()

'''
Purpose: Reads csv file into arrays for plotting and graphs results
'''
def graphNet():
	with open('savedData', newline='') as csvfile:
		netReader = csv.reader(csvfile, delimiter=',')
		for row in netReader:
			hold = list(row)
			if len(row) > 0:
				num_layers_arr.append(row[0])
				num_subconnected_arr.append(row[1])
				num_nodes_arr.append(row[2])
				num_subconnected_nodes_arr.append(row[3])
				batch_size_arr.append(row[4])
				iterations_arr.append(row[5])
				learning_rate_arr.append(row[6])
				window_size_arr.append(row[7])
				error_arr.append(row[8])
				runtime_arr.append(row[9])
				data_size_arr.append(row[10])
	x = error_arr
	x_label = 'Error'
	
	# Graph showing how the number of layers affects error
	plt.figure(1)	
	y = num_layers_arr
	plot_name = 'Error vs Number of Layers'
	y_label = 'Number of Layers'
	graphResults(x, y, plot_name, x_label, y_label)
	
	# Graph showing how the number of subconnected layers affects error
	plt.figure(2)
	y = num_subconnected_arr
	plot_name = 'Error vs Number of Subconnected Layers'
	y_label = 'Subconnected Layers'
	graphResults(x, y, plot_name, x_label, y_label)
	
	# Graph showing how the number of nodes per layer affects error
	plt.figure(3)
	y = num_nodes_arr
	plot_name = 'Error vs Nodes per Layer'
	y_label = 'Nodes per Layer'
	graphResults(x, y, plot_name, x_label, y_label)	
	
	# Graph showing how the number of nodes per subconnected layer affects error
	plt.figure(4)
	y = num_subconnected_nodes_arr
	plot_name = 'Error vs Nodes per Subconnected Layer'
	y_label = 'Nodes per Subconnected Layer'
	graphResults(x, y, plot_name, x_label, y_label)
	
	# Graph showing how the batch size affects error
	plt.figure(5)
	y = batch_size_arr
	plot_name = 'Error vs Batch Size'
	y_label = 'Layers'
	graphResults(x, y, plot_name, x_label, y_label)
	
	# Graph showing how the number of iterations affects error
	plt.figure(6)
	y = iterations_arr
	plot_name = 'Error vs Number of Iterations'
	y_label = 'Iterations'
	graphResults(x, y, plot_name, x_label, y_label)
	
	# Graph showing how the learning rate affects error
	plt.figure(7)
	y = learning_rate_arr
	plot_name = 'Error vs Learning Rate'
	y_label = 'Learning Rate'
	graphResults(x, y, plot_name, x_label, y_label)
	
	# Graph showing how the window size affects error
	plt.figure(8)
	y = window_size_arr	
	plot_name = 'Error vs Window Size'
	y_label = 'Window Size'
	graphResults(x, y, plot_name, x_label, y_label)

	# Graph showing how the number of iterations affects error
	plt.figure(9)
	y = iterations_arr
	plot_name = 'Error vs Number of Iterations'
	y_label = 'Number of Iterations'
	graphResults(x, y, plot_name, x_label, y_label)
	
	# Graph showing how the data_size affects runtime
	plt.figure(10)
	x = data_size_arr
	y = iterations_arr
	plot_name = 'Data Size vs Runtime'
	x_label = 'Data Size'
	y_label = 'Runtime'
	graphResults(x, y, plot_name, x_label, y_label)
