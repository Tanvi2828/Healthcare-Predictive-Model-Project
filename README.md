# **Healthcare Predictive Model: Diabetes Prediction**

## **Project Overview**
This project aims to develop a machine learning model that predicts whether a patient is likely to have diabetes based on health metrics and medical data. The dataset used is the **Pima Indians Diabetes Dataset**, which includes features like glucose level, BMI, and age. This project demonstrates data preprocessing, exploratory data analysis (EDA), and model training for classification tasks in healthcare.

---

## **Key Features**
- Predicts diabetes occurrence with a focus on model accuracy and interpretability.
- Uses Random Forest for classification and feature importance analysis.
- Includes data cleaning, preprocessing, and visualization steps.
- Provides insights into the most significant health metrics affecting diabetes prediction.

---

## **Dataset**
The dataset used is the **Pima Indians Diabetes Dataset**, publicly available on Kaggle and other platforms.  
**Source**: [Kaggle: Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Rows**: 768  
- **Columns**: 9  
  - Features: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
  - Target: `Outcome` (1 = Diabetes, 0 = No Diabetes)

---

## **Steps Performed**
### 1. Data Loading and Exploration
- Loaded the dataset into a Pandas DataFrame.
- Explored the dataset to understand its structure, detect missing values, and examine distributions.

### 2. Data Preprocessing
- Replaced zero values in critical columns (e.g., `Glucose`, `BMI`) with the median of the respective feature.
- Standardized the features using `StandardScaler` for better model performance.

### 3. Model Development
- Split the dataset into training (80%) and testing (20%) sets.
- Used the **Random Forest Classifier** for prediction due to its robustness and feature interpretability.

### 4. Model Evaluation
- Evaluated the model using accuracy, precision, recall, and F1 score.
- Visualized the confusion matrix to analyze prediction performance.

### 5. Insights and Visualization
- Identified the most important features affecting diabetes prediction using feature importance plots.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: 
  - `Pandas`, `NumPy` for data manipulation
  - `Matplotlib`, `Seaborn` for visualization
  - `Scikit-learn` for machine learning and evaluation metrics

---

## **Results**
- **Accuracy**: Achieved ~75-80% accuracy on the test dataset.
- **Key Features**:
  - Glucose
  - BMI
  - Age

### **Confusion Matrix**
Visualized the confusion matrix to understand the classification results:
![Confusion Matrix](link_to_image_placeholder)

### **Feature Importance**
The top features influencing diabetes prediction:
- Glucose: ~40% importance
- BMI: ~20% importance
- Age: ~10% importance

---

## **How to Run the Project**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/healthcare-predictive-model.git
   ```
2. Navigate to the project folder:
   ```bash
   cd healthcare-predictive-model
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python healthcare_predictive_model.py
   ```

---

## **Future Work**
- Experiment with other classification algorithms (e.g., Logistic Regression, XGBoost).
- Hyperparameter tuning for improved accuracy.
- Explore feature engineering techniques to enhance prediction performance.

---

## **Author**
**Tanvi Ghodke**  
*Aspiring AI and Data Science Specialist*

---

## **Acknowledgements**
- Dataset provided by the National Institute of Diabetes and Digestive and Kidney Diseases.
- Inspired by the use of AI in healthcare applications to improve early diagnosis and treatment planning.

---

## **License**
This project is licensed under the MIT License.
