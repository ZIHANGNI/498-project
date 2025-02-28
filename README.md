# Generating Synthetic Climate Data with Deep Learning

## Overview  
This project explores the use of deep learning techniques to generate synthetic climate data conditioned on geographic coordinates. The goal is to create realistic temperature and rainfall patterns for regions with limited or no weather station data. 

By leveraging historical climate records, the model learns spatial dependencies and climate variations, providing valuable synthetic data for climate research, risk assessment, and environmental analysis.

## Dataset   
The project utilizes the **[Climate Change: Earth Surface Temperature dataset](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)** from Kaggle. 
﻿ 
### Data Preprocessing Steps   
- Handling missing values using interpolation techniques   
- Normalizing temperature and rainfall values for better model performance   
- Encoding geographic coordinates as model inputs  

## Methodology   
### 1. Data Processing   
- Extract and clean climate records   
- Normalize temperature and rainfall values   
- Structure data to include geographic coordinates as conditional inputs   
﻿ 
### 2. Model Architecture   
The model follows a **Conditional Generative Adversarial Network** approach:   
- **Generator:** Maps geographic coordinates to synthetic climate data   
- **Discriminator:** Distinguishes real climate data from generated samples   
- **Conditional Input:** Ensures generated climate data is location-specific   
﻿ 
### 3. Training Process   
- Train the model using historical climate records   
- Optimize with adversarial loss and spatial consistency metrics   
- Evaluate generated data by comparing distributions with actual climate records   
﻿
