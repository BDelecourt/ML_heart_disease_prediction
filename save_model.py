# model_report.py
import pandas as pd
from sklearn.metrics import classification_report
import os

def save_model_report(model_name, y_test, y_pred, csv_file_path):
    """
    Saves or updates the classification report of a model to a CSV file.
    
    Parameters:
    - model_name: str, the name of the model (e.g., 'Logistic Regression')
    - y_test: array-like, true labels
    - y_pred: array-like, predicted labels
    - csv_file_path: str, path to the CSV file where results will be saved
    """
    # Get the classification report as a dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Extract metrics for class 0, class 1, and averages
    report = {
        'model': [model_name],  # Model name
        'precision_0': [report_dict['0']['precision']],
        'recall_0': [report_dict['0']['recall']],
        'f1_score_0': [report_dict['0']['f1-score']],
        'precision_1': [report_dict['1']['precision']],
        'recall_1': [report_dict['1']['recall']],
        'f1_score_1': [report_dict['1']['f1-score']],
        'accuracy': [report_dict['accuracy']],
        'macro_avg_precision': [report_dict['macro avg']['precision']],
        'macro_avg_recall': [report_dict['macro avg']['recall']],
        'macro_avg_f1': [report_dict['macro avg']['f1-score']],
        'weighted_avg_precision': [report_dict['weighted avg']['precision']],
        'weighted_avg_recall': [report_dict['weighted avg']['recall']],
        'weighted_avg_f1': [report_dict['weighted avg']['f1-score']]
    }

    # Check if the file exists
    if os.path.exists(csv_file_path):
        # Load the existing CSV file
        df_existing = pd.read_csv(csv_file_path)

        # Check if the model already exists in the CSV
        if model_name in df_existing['model'].values:
            # Update the row corresponding to the model
            df_existing.loc[df_existing['model'] == model_name, ['precision_0', 'recall_0', 'f1_score_0',
                                                                  'precision_1', 'recall_1', 'f1_score_1',
                                                                  'accuracy', 'macro_avg_precision', 'macro_avg_recall',
                                                                  'macro_avg_f1', 'weighted_avg_precision',
                                                                  'weighted_avg_recall', 'weighted_avg_f1']] = \
                report['precision_0'][0], report['recall_0'][0], report['f1_score_0'][0], \
                report['precision_1'][0], report['recall_1'][0], report['f1_score_1'][0], \
                report['accuracy'][0], report['macro_avg_precision'][0], report['macro_avg_recall'][0], \
                report['macro_avg_f1'][0], report['weighted_avg_precision'][0], \
                report['weighted_avg_recall'][0], report['weighted_avg_f1'][0]

            # Save the updated DataFrame back to the CSV
            df_existing.to_csv(csv_file_path, index=False)
            print(f"Updated results for {model_name} in {csv_file_path}")
        else:
            # Append new row if model is not present
            df_new = pd.DataFrame(report)
            df_existing = pd.concat([df_existing, df_new], ignore_index=True)
            df_existing.to_csv(csv_file_path, index=False)
            print(f"Added new results for {model_name} to {csv_file_path}")
    else:
        # If the file does not exist, create a new one with the header
        df = pd.DataFrame(report)
        df.to_csv(csv_file_path, mode='w', header=True, index=False)
        print(f"Saved results to {csv_file_path}")
