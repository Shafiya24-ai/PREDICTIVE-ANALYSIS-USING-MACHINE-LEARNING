# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

**COMPANY** : CODETECH IT SOLUTIONS

**NAME** : SHAIK SHAFIYA

**INTERN ID** : CT06DF1367

**DOMAIN** : DATA ANALYTICS

**DURATION** : 6 WEEKS

**MENTOR** : NEELA SANTHOSH KUMAR

# Breast Cancer Classification using Random Forest Classifier

This project focuses on the development and evaluation of a machine learning model designed to predict the presence of breast cancer in patients using clinical and diagnostic features. Utilizing the Random Forest Classifier, the model analyzes a variety of tumor characteristics to distinguish between malignant and benign tumors. This predictive model demonstrates how machine learning can contribute meaningfully to early diagnosis and medical decision support.

# Data Preparation and Preprocessing

The project utilizes the Breast Cancer Wisconsin Diagnostic dataset, which is a publicly available and well-known dataset in the machine learning community, accessible via the scikit-learn library. This dataset includes 30 numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses. Each record also includes a target label: 0 for malignant and 1 for benign tumors.

Upon loading the dataset, an initial exploratory step was performed using df.head() to preview the structure and contents. A correlation heatmap was generated using seaborn to examine the relationships between features. The visualization helped in understanding how strongly certain features are interrelated, which can inform feature engineering decisions, although no features were dropped in this case.

Before training, the dataset was divided into features (X) and target (y). Since feature scales varied significantly, a StandardScaler was applied to normalize the data, ensuring better performance of the classifier. The dataset was then split into training and testing sets in an 80:20 ratio using train_test_split.

# Feature Selection

All 30 original features were retained in this analysis. A correlation analysis revealed strong inter-feature relationships (e.g., between mean radius and mean area), but no feature was considered redundant enough to be removed. Additionally, a feature importance plot derived from the Random Forest model was later used to evaluate which features contributed most to the prediction. Notably, features like mean concave points, mean perimeter, and mean area were found to be highly influential.

# Model Training and Evaluation

A Random Forest Classifier with 100 trees (n_estimators=100) and a fixed random state was trained on the normalized training data. Random Forest is an ensemble method that builds multiple decision trees and combines their outputs to improve classification accuracy and reduce overfitting.

Once trained, the model was evaluated on the test set. The following metrics were used to assess its performance:

- **Accuracy Score:** The model achieved an accuracy of approximately 96%, reflecting strong predictive capability.

- **Classification Report:** Precision, recall, and F1-score were computed for both classes (malignant and benign). The F1-scores were high for both categories, confirming a balanced performance.

- **Confusion Matrix:** A heatmap of the confusion matrix was plotted for better visual understanding of false positives and false negatives.

The classification report and confusion matrix indicate the model’s robustness in handling both types of tumors with minimal misclassifications.

### Tools and Libraries Used

- **Programming Language:** Python

- **Data Analysis:** Pandas, NumPy

- **Visualization:** Matplotlib, Seaborn

- **Machine Learning:** Scikit-learn (RandomForestClassifier, preprocessing, model evaluation)

- **Development Environment:** Jupyter Notebook 

# Dataset Description

The dataset used is the Breast Cancer Wisconsin Diagnostic Dataset from the UCI Machine Learning Repository, accessible via sklearn.datasets.load_breast_cancer(). It includes 569 samples with 30 numerical features and a binary target indicating whether a tumor is malignant (0) or benign (1). These features represent mean, standard error, and worst values of tumor properties such as radius, texture, smoothness, concavity, etc.

### Real-World Applications

**Early Diagnosis:** Supports clinicians in identifying potential cancer cases at early stages.

**Screening Tools:** Can be used in digital mammography or pathology software to highlight high-risk cases.

**Decision Support Systems:** Assists in clinical decision-making alongside radiologists and oncologists.

**Educational Use:** Serves as an excellent teaching example for supervised learning in healthcare.

**AI-Assisted Diagnostics:** Forms the foundation for building intelligent diagnostic assistants.

# Conclusion

This project showcases how Random Forest, a powerful ensemble learning technique, can be effectively applied to medical diagnostic data to predict breast cancer outcomes with high accuracy. While the results are promising, future improvements can be made by experimenting with other classifiers, hyperparameter tuning, and integrating domain-specific knowledge into feature engineering. The project serves as a strong demonstration of how data-driven models can support healthcare professionals in making informed, life-saving decisions.



