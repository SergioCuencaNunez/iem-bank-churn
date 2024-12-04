# Customer Segmentation and Credit Card Analysis

This project focuses on the **analysis and segmentation of credit card customers** for a banking institution. The goal is to predict customer churn and develop actionable insights to enhance customer retention, group clients based on shared characteristics, and design personalized loyalty programs.

## Objectives

1. **Churn Prediction**: Identify customers likely to leave the bank.
2. **Customer Grouping**: Cluster clients with similar attributes for targeted analysis.
3. **Loyalty Programs**: Design custom strategies to retain customers.
4. **Service Quality Improvement**: Use data insights to enhance customer experience.

## Methodologies and Models Used

### 1. **Preprocessing**
   - Exclusion of irrelevant or highly correlated variables.
   - Grouping data (e.g., merging single and divorced clients).
   - Transformation using:
     - **Power Transform**
     - **Standard Scaler**
     - **Min-Max Scaler**
     - **One-Hot Encoding**

### 2. **Association Rules**
   - **Apriori Algorithm** with:
     - Support ≥ 35%
     - Confidence ≥ 90%
     - Lift ≥ 1.3
   - Key insights:
     - Gender is linked to income levels.
     - Low-income customers mostly own basic credit cards.

### 3. **Decision Trees (CART)**
   - Built classification trees.
   - Identified `Total_Trans_Ct`, `Total_Trans_Amt`, and `Total_Revolving_Bal` as key predictors.

### 4. **Random Forest**
   - Enhanced handling of imbalanced datasets.
   - Achieved high precision for identifying churned customers.
   - Optimal hyperparameters determined via **Out-of-Bag Error**.

### 5. **Boosting (Gradient Boosting)**
   - Iterative improvement targeting misclassified instances.
   - Superior recall and F1-scores compared to other methods.

### 6. **Clustering**
   - Performed hierarchical clustering using the **Ward method**.
   - Segmented customers into clusters based on socioeconomic attributes and churn likelihood.

### 7. **Neural Networks**
   - Used TensorFlow for **MLP (Multi-Layer Perceptron)** models.
   - Fine-tuned for faster training and higher performance on unseen data.


## Key Findings

- **Churn Indicators**: Variables like `Total_Trans_Ct` and `Total_Trans_Amt` consistently influence predictions across models.
- **Cluster Profiles**:
  - High-income customers require personalized investment services.
  - Low-income segments benefit from debt management tools.
  - Medium-income clusters value financial stability programs.
- **Model Recommendations**:
  - Random Forest and Boosting outperform other models in churn detection.
  - Neural networks, while accurate, are more prone to false negatives.

## Results

| Metric         | CART  | Random Forest | Boosting | Neural Networks |
|----------------|-------|---------------|----------|-----------------|
| Precision      | 97%   | 97%           | 98%      | 94%             |
| Recall         | 94%   | 99%           | 98%      | 93%             |
| F1-Score       | 96%   | 98%           | 98%      | 96%             |
| Accuracy       | 93%   | 96%           | 97%      | 93%             |

## Tools and Technologies

- **Languages**: Python (TensorFlow, scikit-learn)
- **Libraries**: pandas, NumPy, matplotlib, seaborn
- **Algorithms**: CART, Random Forest, Gradient Boosting, Neural Networks
- **Data Preprocessing**: Scalers, One-Hot Encoding, Apriori

## Conclusion

This analysis demonstrates the effective use of machine learning and clustering techniques to understand customer behavior. By leveraging these insights, banks can optimize loyalty programs and proactively address customer needs, reducing churn and enhancing customer satisfaction.

---

For any queries or contributions, please contact **Elena Conderana** or **Sergio Cuenca**.
