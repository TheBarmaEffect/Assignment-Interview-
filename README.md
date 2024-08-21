# **Loan Application Prediction Report**

## **1. Approach Taken**

### **1.1 Initial Steps and Challenges**

#### **1.1.1 Data Loading and Exploration**

The project began with loading the training and test datasets, which contained information on two-wheeler loan applications. The training data included the target variable (`Application Status`), while the test data did not. Initial exploration was focused on understanding the structure of the datasets, identifying the presence of missing values, categorical variables, and date fields, and assessing class distribution.

- **Observation 1:** There were significant missing values across several features, particularly in both categorical and numerical columns.
- **Observation 2:** The data contained multiple categorical variables with potentially high cardinality (e.g., location codes, branch codes).
- **Observation 3:** Date fields (e.g., `APPLICATION LOGIN DATE`, `DOB`) were present, but their formats were inconsistent, raising concerns about their utility without proper conversion.

#### **1.1.2 Initial Preprocessing Attempts**

The initial preprocessing steps included basic handling of missing values and encoding categorical variables. However, multiple challenges emerged:

- **Handling Missing Values:**
  - **Issue:** Applying `SimpleImputer` to both numerical and categorical data led to inconsistent results. The strategy used (`mean` for numerical and `most_frequent` for categorical) seemed straightforward, but issues arose when certain categorical values present in the test data were not seen in the training data.
  - **Result:** This led to `KeyError` and `ValueError` during the initial model prediction phase, primarily due to unseen labels in the test set.

- **Categorical Encoding Issues:**
  - **Issue:** The initial attempt to use `LabelEncoder` caused problems, particularly with unseen categories in the test set. Since `LabelEncoder` only encodes categories seen during training, this resulted in errors when making predictions on the test set.
  - **Result:** The model failed to handle these cases, leading to an inability to make predictions.

- **Date Feature Conversion:**
  - **Issue:** Date fields such as `APPLICATION LOGIN DATE` and `DOB` were initially in inconsistent formats. Attempts to directly convert these dates into numerical features failed due to format mismatches.
  - **Result:** The conversion process led to errors, and the initial model did not benefit from these potentially valuable time-based features.

### **1.2 Steps Taken to Overcome Challenges**

#### **1.2.1 Advanced Handling of Missing Values**

To address the issues with missing values:

- **Numerical Columns:** 
  - **Approach:** Applied `SimpleImputer` with the `mean` strategy to all numerical columns. This ensured that missing numerical values were consistently handled across both training and test datasets.
  - **Impact:** This reduced the variability caused by missing values in numerical features and ensured that the model was not skewed by these gaps.

- **Categorical Columns:**
  - **Approach:** Used `SimpleImputer` with the `most_frequent` strategy for categorical columns. This was coupled with a custom strategy for encoding unseen labels. Specifically, a fallback value was introduced to handle categories that appeared in the test data but not in the training set.
  - **Impact:** This approach ensured that the model could still make predictions even when encountering previously unseen categories, thus improving robustness.

#### **1.2.2 Date Feature Engineering**

Given the initial failures in handling date fields, a more sophisticated approach was taken:

- **Standardizing Dates:**
  - **Approach:** Used `pd.to_datetime` with `infer_datetime_format=True` and `errors='coerce'` to handle inconsistencies in date formats. This allowed for the extraction of meaningful features such as year, month, and day from the date fields.
  - **Impact:** These new features captured temporal patterns and trends that could influence loan approval decisions, adding depth to the model's understanding.

- **Feature Extraction:**
  - **Approach:** Features such as `APPLICATION LOGIN YEAR`, `APPLICATION LOGIN MONTH`, and `APPLICATION LOGIN DAY` were created. Additionally, differences in time, such as the applicant’s age at the time of application (derived from `DOB`), were calculated.
  - **Impact:** These derived features provided the model with insights into seasonality, application trends over time, and the applicant’s age, all of which were critical factors in loan approval decisions.

#### **1.2.3 Robust Categorical Encoding**

To address the categorical encoding issues:

- **Safe Encoding with LabelEncoder:**
  - **Approach:** Implemented a safer encoding strategy using `LabelEncoder` with fallback handling for unseen labels. Specifically, if a label was not seen during training, it was assigned a special value (e.g., -1), allowing the model to handle such cases without error.
  - **Impact:** This strategy ensured that the model could make predictions even on data with categories it had not encountered during training, thus improving its generalization capability.

- **One-Hot Encoding for High Cardinality:**
  - **Approach:** For categorical features with high cardinality (many unique values), One-Hot Encoding was used. This approach prevented the model from being overwhelmed by a single categorical feature and ensured that each category was treated independently.
  - **Impact:** This reduced the risk of overfitting to specific categories and ensured that the model could handle a wide range of inputs.

#### **1.2.4 Model Selection and Cross-Validation**

Given the complexity of the problem, multiple models were evaluated:

- **Model Selection:**
  - **Models Tested:** Random Forest, Gradient Boosting, Logistic Regression, and SVM.
  - **Approach:** Each model was integrated into a pipeline that included preprocessing steps (handling missing values, encoding, feature scaling) and was evaluated using 5-fold cross-validation.
  - **Impact:** This approach ensured that each model was rigorously tested across multiple subsets of the data, providing a robust evaluation of their generalization performance.

- **Model Performance:**
  - **Outcome:** Logistic Regression emerged as the best model, achieving the highest cross-validation accuracy of 0.8632. This indicated that the relationships in the data were largely linear, which Logistic Regression could capture effectively.

#### **1.2.5 Hyperparameter Tuning**

Once Logistic Regression was identified as the best model, further optimization was pursued:

- **GridSearchCV:**
  - **Approach:** Applied `GridSearchCV` to fine-tune hyperparameters such as regularization strength (`C`) and solver choice (e.g., `liblinear`, `lbfgs`).
  - **Impact:** Hyperparameter tuning led to a modest improvement in the model’s performance, confirming that the model was well-optimized for the given dataset.

- **Results:**
  - **Outcome:** The tuned Logistic Regression model showed slightly better performance on the validation set, reinforcing its selection as the final model.

#### **1.2.6 Validation and Final Model Selection**

The final step involved validating the model on a hold-out validation set:

- **Hold-Out Validation:**
  - **Approach:** The tuned Logistic Regression model was evaluated on a separate validation set that was not used during training or cross-validation.
  - **Impact:** The validation accuracy was 0.865, which was consistent with the cross-validation results, indicating that the model generalizes well to unseen data.

- **Final Metrics:**
  - **Outcome:** The model demonstrated strong performance, particularly in predicting the majority class (approved loans). However, it also showed reasonable performance in predicting the minority class (declined loans), with a recall of 0.84.

### **1.3 Reflection and Additional Considerations**

#### **1.3.1 Feature Importance and Engineering**

- **Interaction Terms:**
  - **Potential Approach:** Creating interaction terms between features (e.g., product of two features) could help the model capture non-linear relationships that were not immediately apparent.
  - **Potential Impact:** This could improve the model’s ability to differentiate between subtle differences in loan applications, particularly for borderline cases.

- **Polynomial Features:**
  - **Potential Approach:** Introducing polynomial features (e.g., squares or higher-order terms of existing features) could help the model capture more complex relationships.
  - **Potential Impact:** This would be particularly useful for improving the model’s performance on the minority class, where linear relationships may not fully capture the decision boundary.

- **Target Encoding:**
  - **Potential Approach:** For high-cardinality categorical features, target encoding (replacing categories with the mean of the target variable for each category) could be explored.
  - **Potential Impact:** This would allow the model to leverage the relationship between categorical features and the target variable more effectively, particularly when there are too many categories for One-Hot Encoding to be feasible.

#### **1.3.2 Handling Class Imbalance**

- **SMOTE (Synthetic Minority Over-sampling Technique):**
  - **Potential Approach:** Applying SMOTE to the training data could help balance the class distribution by synthetically generating new instances of the minority class.
  - **Potential Impact:** This could improve the model’s recall for the minority class (declined loans), reducing the number of false negatives.

- **ADASYN (Adaptive Synthetic Sampling):**
  - **Potential Approach:** Similar to SMOTE, but more focused on generating synthetic data points for the minority class where it is most needed (i.e., near the decision boundary).
  - **Potential Impact:** This would further enhance the model’s ability to correctly identify declined loans, particularly in cases where the decision is not straightforward.

- **Cost-Sensitive Learning:**
  - **Potential Approach:** Adjusting the model to assign higher penalties to misclassifications of the minority

- **Cost-Sensitive Learning:**
  - **Potential Approach:** Adjusting the model to assign higher penalties to misclassifications of the minority class (declined loans). This can be done by altering the loss function or applying class weights during model training.
  - **Potential Impact:** This would help the model to focus more on correctly predicting the minority class, potentially improving precision and recall for declined loans.

- **Threshold Adjustment:**
  - **Potential Approach:** Instead of relying on the default decision threshold (usually 0.5), adjusting the threshold to favor the minority class could help improve its recall. For example, lowering the threshold might increase the number of correctly identified declined loans.
  - **Potential Impact:** This could improve the model's sensitivity to the minority class, though it may also increase false positives in the majority class.

#### **1.3.3 Further Model Exploration**

- **Ensemble Methods:**
  - **Potential Approach:** Leveraging ensemble techniques such as stacking, voting, or blending could combine the strengths of multiple models. For instance, combining the predictions of Logistic Regression, Random Forest, and SVM could result in better overall performance.
  - **Potential Impact:** Ensemble methods often yield better generalization than single models by reducing overfitting and capturing a wider range of patterns in the data.

- **Neural Networks and Deep Learning:**
  - **Potential Approach:** Given a larger dataset, neural networks or deep learning models could be explored. These models can automatically learn complex feature interactions and non-linear relationships without manual feature engineering.
  - **Potential Impact:** Deep learning models, particularly when trained on extensive datasets, could potentially outperform traditional models in capturing the underlying patterns in loan approval decisions, especially if more data were available.

- **Gradient Boosting Machines (GBMs):**
  - **Potential Approach:** While Gradient Boosting was tested, more advanced variants like XGBoost, LightGBM, or CatBoost could be explored. These models are known for their efficiency and effectiveness in handling structured/tabular data.
  - **Potential Impact:** These models could offer better performance, particularly in scenarios with high-dimensional feature spaces and complex interactions.

## **2. Insights and Conclusions from Data**

### **2.1 Key Insights**

- **Model Performance:** Logistic Regression emerged as the best model with a cross-validation accuracy of 0.8632. The model's success suggests that the relationships in the data are largely linear, which this model captures effectively.
- **Class Imbalance:** The data exhibited a moderate class imbalance, with more loans being approved than declined. The model managed to maintain good precision and recall across both classes, though it naturally performed better on the majority class.
- **Feature Importance:** The importance of robust preprocessing became evident. Proper handling of categorical variables, date features, and missing values was crucial to the model's success. Additionally, derived features such as age and application month/year provided valuable insights that improved model performance.

### **2.2 Conclusions**

- **Model Selection:** Logistic Regression was selected as the final model due to its high accuracy and generalization ability. Despite its simplicity, it was able to outperform more complex models, likely due to the linear nature of the relationships in the data.
- **Feature Engineering:** The derived features from date fields and careful encoding of categorical variables were significant contributors to the model’s performance. However, there is potential to further enhance these features, particularly by exploring interaction terms, polynomial features, and target encoding.
- **Handling Imbalance:** While the model performed well, especially in predicting approved loans, there remains room for improvement in predicting declined loans. Techniques such as SMOTE, cost-sensitive learning, or adjusting decision thresholds could help address this.
- **Future Work:** There is a clear path for further improvement by exploring advanced models, ensemble techniques, and deeper feature engineering. Additionally, if more data were available, particularly more declined loan applications, deep learning models could be an avenue worth exploring.

## **3. Performance on Train Data Set**

### **3.1 Cross-Validation Results:**
- **Random Forest:** 0.8427
- **Gradient Boosting:** 0.8442
- **Logistic Regression:** 0.8632
- **Support Vector Machine:** 0.8574

### **3.2 Best Model:**
- **Logistic Regression:** With an accuracy of 0.8632 during cross-validation and 0.865 on the validation set, this model demonstrated strong generalization capabilities.

### **3.3 Validation Metrics:**
- **Accuracy:** 0.865
- **Confusion Matrix:**
[[1163 164]
[ 106 567]]

- **Precision, Recall, and F1-Score:**
- **Approved Loans:** Precision: 0.92, Recall: 0.88, F1-Score: 0.90
- **Declined Loans:** Precision: 0.78, Recall: 0.84, F1-Score: 0.81
- **Macro Average F1-Score:** 0.85

### **3.4 Final Prediction Summary:**
- **Approved Loans:** 1277 predicted
- **Declined Loans:** 723 predicted

## **4. Use of Appropriate Metrics**

### **4.1 Accuracy**
- **Description:** Accuracy provided a general measure of how many loan applications were correctly classified by the model. It was the primary metric used to evaluate model performance during cross-validation.
- **Result:** The final model achieved an accuracy of 0.865 on the validation set, indicating that the majority of predictions were correct.

### **4.2 Precision and Recall**
- **Description:** 
- **Precision:** Precision measured the accuracy of the positive predictions (approved loans). It was particularly important in assessing how many of the loans predicted as approved were actually approved.
- **Recall:** Recall measured the model's ability to identify all positive instances. For the minority class (declined loans), recall was crucial in determining how many declined loans were correctly identified.
- **Result:** 
- **Approved Loans:** Precision: 0.92, Recall: 0.88
- **Declined Loans:** Precision: 0.78, Recall: 0.84

### **4.3 F1-Score**
- **Description:** The F1-Score provided a balance between precision and recall. It was particularly useful for evaluating the model's performance on imbalanced classes.
- **Result:** The model achieved a macro average F1-Score of 0.85, indicating a balanced performance across both classes.

### **4.4 Confusion Matrix**
- **Description:** The confusion matrix offered a detailed breakdown of the model's performance, showing the true positives, true negatives, false positives, and false negatives. This allowed for a deeper understanding of where the model was performing well and where it could improve.
- **Result:** The confusion matrix revealed that while the model was generally accurate, there were still instances of misclassification, particularly in predicting declined loans.

## **Conclusion**

The journey through this project highlighted the importance of robust data preprocessing, careful model selection, and continuous iteration to address challenges as they arose. The final Logistic Regression model performed admirably, particularly considering the linear nature of the relationships in the data. However, there is still potential for improvement, particularly in handling class imbalance and capturing more complex relationships between features.

### **Future Work:**
- **Advanced Feature Engineering:** Future work could involve exploring more advanced feature engineering techniques, such as interaction terms, polynomial features, and target encoding.
- **Handling Imbalance:** Techniques like SMOTE, ADASYN, and cost-sensitive learning should be explored further to improve the model's performance on the minority class.
- **Model Exploration:** Experimenting with ensemble methods, deep learning, and advanced gradient boosting machines could yield better results, especially if more data becomes available.


