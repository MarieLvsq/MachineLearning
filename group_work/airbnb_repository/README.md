# Airbnb Busyness Recommendation Engine

Analysis:   
[![Open Analysys In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tm3-machine-learning/airbnb/blob/main/final-analysis/airbnb_analysis.ipynb)  
Modelling:   
[![Open Moddeling In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tm3-machine-learning/airbnb/blob/main/final-analysis/airbnb_modelling.ipynb)

This repository contains the code and resources for a “Airbnb Busyness Recommendation Engine” project. The goal of this project is to demonstrate how Airbnb (or a similar platform) could leverage listing‐level data to measure neighbourhood “busyness” and then recommend underutilised (less busy) but similar neighbourhoods when a user’s desired area is at capacity. By redirecting guest demand to quieter areas, the system aims to help mitigate overtourism and make better use of available inventory.

## Authors
- Bowyer, Matthew
- Chandler, Mark
- Contardi, Thiago
- Levesque, Marie  

## Overview

When a popular neighbourhood becomes crowded, many guests struggle to find alternative—but still appealing—options nearby. This repository implements a proof‐of‐concept K-Nearest Neighbors (KNN) recommendation engine that:

1. **Quantifies “Busyness”**  

2. **Builds Neighbourhood Profiles**

3. **Recommends Less Busy Neighbourhoods**

4. **Provides Transparency and Interpretability**


**Serves as a Proof-of-Concept**  

Although not yet optimised for production, the code demonstrates how listing‐level and aggregated neighbourhood data can be combined to address overtourism in a data‐driven way.  

## Dataset 
The dataset used in this project is a subset of Airbnb listings in New York City, the dataset is available at [Airbnb NYC Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

