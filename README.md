These are three websites made with the Streamlit Library that are meant to be a companion to my paper "Impact of Dimensionality Reduction on Outlier Detection: an Empirical Study", which can be found at: https://ieeexplore.ieee.org/abstract/document/10063503.
Each uses a different method of dimensional reduction, links to access them are below.

PCA: https://outlierdetectioninsmallerdimensions-pca.streamlit.app

Random Projection:  https://outlierdetectioninsmallerdimensions-randomprojection.streamlit.app

UMAP: https://outlierdetectioninsmallerdimensions-umap.streamlit.app

This is a link to a video presentation of the paper, with a demonstration of the websites at the end: https://ieeecps.org/files/28VK1c2bFDq8efZDdLHCtW

All of the datasets tested in the paper are from the ODDS (Outlier Detection Datasets) repository provided by Stonybrook University, which can be found at: http://odds.cs.stonybrook.edu

If you wish to input your own data, it needs to be in .mat format, and needs to become a dictionary with two keys, "X" and "y" when read into Scipy.io. X is the data itself, and y is labels for X. 0 means that that point in an inlier, and 1 means it's an outlier.

If you want to run any of these websites offline, make sure everything in requirements.txt is installed and then download that website's respective file, Outlier_Project_functions.py, and lof.py. Make sure all three are in the same directory. To run the file, type "streamlit run [file path of the website file]" in the console.
