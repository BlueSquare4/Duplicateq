Project Summary: Duplicate Question Checker for Quora Dataset
Objective:
To develop a robust model capable of detecting duplicate questions in the Quora dataset, thereby improving user experience by reducing redundant content.
Key Contributions:
Data Collection and Preprocessing:


Utilized a dataset containing 400K Quora question pairs.
Preprocessed text data by removing stop words, punctuation, and applying tokenization and stemming.
Feature Engineering:


Engineered 7 basic features including word overlap, common word ratio, and length difference.
Added 15 advanced features such as TF-IDF cosine similarity and Jaccard similarity.
Integrated Fuzzy features using the FuzzyWuzzy library to capture partial and token-based similarity.
Dimensionality Reduction and Visualization:


Applied t-SNE for dimensionality reduction to visualize high-dimensional data.
Used Plotly and Seaborn libraries to create insightful visualizations of question pair distributions.
Model Development:


Experimented with various machine learning models, including Logistic Regression, Random Forest, and XGBoost.
Achieved a maximum accuracy of 80.82% with feature-engineered data using Random Forest and XGBoost classifiers.
Results:
The final model achieved an accuracy of 80.82%, demonstrating its capability to effectively identify duplicate questions.
Feature engineering played a significant role in boosting model performance.
Future Scope:
Incorporate advanced NLP techniques such as BERT or Siamese LSTM networks to further improve accuracy.
Deploy the model as an API for real-time duplicate question detection.
Expand the model to handle multilingual question pairs.
This project highlights the importance of feature engineering and model selection in building high-performing duplicate question detection systems.

