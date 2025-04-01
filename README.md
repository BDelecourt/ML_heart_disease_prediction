# Heart Disease Prediction with Machine Learning

## Project Overview
This project explores different machine learning models to predict heart disease using the `heart.csv` dataset. Each model is tested, and its performance metrics are compiled into `model_comparison.csv`.

## Project Structure
## Project Structure
<pre>
/ML_heart_disease_prediction  
    │── heart.csv # Dataset used for training and testing  
    │── model_comparison.csv # Performance results of different models  
    │── LogisticRegression_model.py # Implements Logistic Regression  
    │── DecisionTree_model.py # Implements Decision Tree Classifier  
    │── RandomForest_model.py # Implements Random Forest Classifier  
    │── GradientBoosting_model.py # Implements Gradient Boosting  
    │── README.md # Project documentation  
</pre>
## Dataset
The dataset `heart.csv` contains medical attributes used to determine the presence of heart disease. It includes features like age, sex, cholesterol levels, and more.

## Models Implemented
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

## Model Performance Summary
| Model                         | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|--------------------------------|----------|------------------|---------------|----------------|
| Logistic Regression            | 80.98%   | 82.40%           | 80.71%        | 80.66%         |
| Decision Tree Classifier       | 98.54%   | 98.54%           | 98.57%        | 98.54%         |
| Random Forest Classifier       | 100.00%  | 100.00%          | 100.00%       | 100.00%        |
| Gradient Boosting Classifier   | 97.56%   | 97.57%           | 97.55%        | 97.56%         |

## Conclusion
The Random Forest Classifier achieved perfect accuracy on this dataset, followed closely by Decision Tree and Gradient Boosting. However, Logistic Regression also performed well and could be preferable in cases requiring interpretability.

## How to Use
1. **Install Dependencies**  
   Ensure you have Python installed, then install the required packages by running:
   ```sh
   pip install -r requirements.txt
   ```
   PS: Use a virtual environnement for clearner development and not interfer with other project
2. **Run the Model Scripts**
    ```sh
    python logistic_regression_model.py
    python decision_tree_model.py
    python random_forest_model.py
    python gradient_boosting_model.py
   ```

3. **Review Model Performance**
    After execution, check model_comparison.csv for a summary of model results.

## Future Work
- Expand dataset for better generalization.
- Hyperparameter tuning for optimized models.
- Explore deep learning approaches.
