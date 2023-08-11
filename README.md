# Gaussian-Biased-Collaborative-Filtering
This project introduces a novel approach, Gaussian Biased Collaborative filtering, to movie recommendation systems using Singular Value Decomposition++ (SVD++) and Restricted Boltzmann Machines (RBM) and the inclusion of Gaussian biases on the MovieLens-100k dataset.


#Author:

Chi Zhang 	
Enze Tao  	
Pengxiao Han	

#Date: 28/05/2023

#Algorithm:	
    Gaussian-Bias.ipynb  
    Normal-Bias.ipynb  
    visualize.py  

#CSV files:	
        Gau_biased_data_n10.csv  
        Gau_biased_data_n20.csv  
        Gau_biased_data_n30.csv  
		    Gau_biased_data_n50.csv  
		    Nor_biased_data_n10.csv  
		     Nor_biased_data_n20.csv    
		    Nor_biased_data_n30.csv  
		    Nor_biased_data_n50.csv  
		    pridicted_ratings.csv  

Gaussian-Bias.ipynb and Normal-Bias.ipynb are code to demonstrate our model, 
which includes data preprocessing, file revised, SVD, SVD++ and RBM.

When you need to run code for different csv, you just revise the variable 'local_file' to make it point to the csv file.

visualize.py use pridiceted_ratings.csv to plot the visualized figure about movies.
