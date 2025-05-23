# Principal Component Analysis (PCA) on Score Dataset

This Python script performs Principal Component Analysis (PCA) on a dataset of numerical scores stored in a text file (`scores`). PCA is a widely used dimensionality reduction technique that helps identify the directions (principal components) along which the data varies the most. This project implements PCA from scratch using fundamental linear algebra tools provided by NumPy and SciPy, and visualizes the results with Matplotlib.

---

## Overview

1. **Mean Centering:**  
   The mean of each feature is calculated and subtracted from the data to center it around zero. This step is critical for PCA to work properly.

2. **Singular Value Decomposition (SVD):**  
   The centered data matrix is normalized and decomposed using SVD to extract principal components and singular values.

3. **Gamma Coefficients Calculation:**  
   A predefined weight vector `g` is projected onto the principal components to compute gamma coefficients, which represent the contribution of each component.

4. **Visualization:**  
   The script generates scatter plots showing relationships between transformed data and weighted components, facilitating intuitive understanding of the principal components’ influence.

---

## Results

- The first principal component has the highest impact on the final score which is represented by the strong, positive correlation in the graph
- The second and third principal compnents both impact the final score, but the affect is much weaker than the first principal component
- The fourth, fifth, and sixth principal components all barel impact the final score, and have extremeley weak correlations in the graph. 

---

## Features

- Manual implementation of PCA using matrix operations and SVD  
- Data preprocessing with mean-centering  
- Interpretation enhancement by adjusting principal component signs  
- Visualization of principal components and their weighted effects on the dataset  
- Reconstruction error computation to verify the accuracy of the PCA approximation  

---

## Requirements

- Python 3.x  
- numpy  
- scipy  
- matplotlib  
