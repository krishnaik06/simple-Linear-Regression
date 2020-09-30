# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Creating regression class

class Linear_regression:
	
	def __init__(self,train_data,train_labels):
		self.train_data = train_data
		self.train_labels = train_labels
		self.new_train_data = np.insert(self.train_data,0,1,axis=1)		
		self.weights = np.zeros((2,1))		
		self.epochs = 1500
		self.alpha = 0.01

	def hypothesis(self):
		return np.dot(self.new_train_data,self.weights)	
		
	def cost(self):
		cost = (1/(2*np.size(self.train_labels)))*np.sum((self.hypothesis()-self.train_labels)**2)
		return cost		

	def derivative(self):
		return (1/np.size(self.train_labels))*np.dot(self.new_train_data.T,(self.hypothesis()-self.train_labels))

	def train(self):
		self.loss = []		
		for i in range(self.epochs):
			cost = self.cost()					
			self.weights = self.weights - (self.alpha) * self.derivative()
			self.loss.append(cost)
		
		plt.plot(self.loss)
		plt.show()	
		return self.weights,np.array(self.loss)
		
	def predict(self,data):
		return np.dot(data,self.weights)	

	def visualize(self,data):
		data = self.hypothesis()
    plt.title('Salary vs Experience')
		plt.xlabel('Years of Experience')
		plt.ylabel('Salary')		
		plt.scatter(self.train_data,self.train_labels,marker='x',color='red',label='Training data')
		plt.plot(self.new_train_data[:,1],data,label='Linear regression')
		plt.legend(loc='lower right')
		plt.show()						

if __name__ == '__main__':
  
  # Reading data   
  
	data = pd.read_csv('Salary_Data.csv')
	train_data = np.array(data.iloc[:,:1])
	train_labels = np.array(data.iloc[:,1:])			

#   Applying Linear regression  
  
	linear = Linear_regression(train_data,train_labels)	
	print('older cost: ',gd.cost())
	result = gd.train()
	print('updated theta: \n',result[0])
	print('final cost: ',gd.cost())
	gd.visualize(gd.hypothesis())	
