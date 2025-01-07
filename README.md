# Duplicateq

This repository contains the implementation of a **Duplicate Question Checker** for the Quora question pairs dataset. The project involves preprocessing text data, feature engineering, and applying machine learning models to detect duplicate questions with high accuracy.

## Project Overview

### Objective
The goal of this project is to develop a model that can effectively identify whether two given questions are duplicates or not. This helps in improving user experience by reducing redundant questions on platforms like Quora.

### Dataset
- **Quora Question Pairs Dataset**
- Contains **400K question pairs** with labels indicating whether the pairs are duplicates.

### Key Features
1. **Data Preprocessing**: Cleaning and tokenizing text data.
2. **Feature Engineering**:
   - 7 basic features (e.g., length difference, common word count)
   - 15 advanced features (e.g., TF-IDF cosine similarity, Jaccard similarity)
   - Fuzzy features using the `FuzzyWuzzy` library.
3. **Dimensionality Reduction**: Applied **t-SNE** for visualization.
4. **Model Training**:
   - Used **Random Forest** and **XGBoost** classifiers.
   - Achieved maximum accuracy of **80.82%**.
5. **Hyperparameter Tuning**: Performed GridSearch to fine-tune the model parameters.

## Results
- Achieved an accuracy of **80.82%**.
- Visualized the high-dimensional data using **t-SNE**, resulting in insightful visualizations.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Duplicateq.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the dataset.
2. Run the preprocessing and feature engineering scripts.
3. Train the model using `Random Forest` or `XGBoost`.
4. Evaluate the model using accuracy and F1 score.
5. Visualize the results using t-SNE.

## Dependencies
- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FuzzyWuzzy
- Matplotlib
- Seaborn
- Plotly

## Folder Structure
```
Duplicateq/
├── data/                 # Dataset (not included)
├── notebooks/            # Jupyter notebooks for EDA and feature engineering
├── main.py                  # Source code
├── README.md             # Project documentation
└── requirements.txt      # List of dependencies
```


## Acknowledgments
- Special thanks to Quora for providing the dataset.
- Inspired by various open-source projects on duplicate question detection.

Feel free to contribute by submitting issues or pull requests!

