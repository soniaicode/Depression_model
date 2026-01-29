"""
Confusion Matrix Visualization Script
Creates confusion matrices for all trained models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_confusion_matrix(cm, model_name, accuracy, save_path):
    """
    Plot confusion matrix with annotations
    
    Args:
        cm: Confusion matrix array [[TN, FP], [FN, TP]]
        model_name: Name of the model
        accuracy: Model accuracy
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar=True, square=True,
                xticklabels=['No Depression (0)', 'Depression (1)'],
                yticklabels=['No Depression (0)', 'Depression (1)'],
                annot_kws={'size': 16, 'weight': 'bold'})
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # Add title with metrics
    plt.title(f'{model_name}\nAccuracy: {accuracy:.2%}', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    
    # Add text annotations for TN, FP, FN, TP
    plt.text(0.5, 0.15, f'TN = {tn}', ha='center', va='center', 
             fontsize=10, color='darkblue', transform=plt.gca().transAxes)
    plt.text(1.5, 0.15, f'FP = {fp}', ha='center', va='center', 
             fontsize=10, color='darkred', transform=plt.gca().transAxes)
    plt.text(0.5, 0.85, f'FN = {fn}', ha='center', va='center', 
             fontsize=10, color='darkred', transform=plt.gca().transAxes)
    plt.text(1.5, 0.85, f'TP = {tp}', ha='center', va='center', 
             fontsize=10, color='darkblue', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_combined_confusion_matrices(results_data, save_path):
    """
    Create a combined figure with all confusion matrices
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion Matrices - All Models', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    models = [
        ('logistic_regression', 'Logistic Regression'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('deep_learning', 'Deep Learning')
    ]
    
    for idx, (model_key, model_name) in enumerate(models):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Get confusion matrix and accuracy
        cm = np.array(results_data[model_key]['confusion_matrix'])
        accuracy = results_data[model_key]['accuracy']
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    cbar=True, square=True, ax=ax,
                    xticklabels=['No Depression', 'Depression'],
                    yticklabels=['No Depression', 'Depression'],
                    annot_kws={'size': 14, 'weight': 'bold'})
        
        # Calculate metrics
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        precision = results_data[model_key]['precision']
        recall = results_data[model_key]['recall']
        f1 = results_data[model_key]['f1_score']
        
        # Set title with metrics
        title = f'{model_name}\n'
        title += f'Acc: {accuracy:.1%} | Prec: {precision:.1%} | Rec: {recall:.1%} | F1: {f1:.1%}'
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        
        ax.set_ylabel('Actual', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def print_confusion_matrix_details(results_data):
    """
    Print detailed confusion matrix information for all models
    """
    print("\n" + "="*80)
    print("CONFUSION MATRIX ANALYSIS - ALL MODELS")
    print("="*80)
    
    for model_key, data in results_data.items():
        model_name = data['model_name']
        cm = np.array(data['confusion_matrix'])
        
        # Extract values
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        
        print(f"\n{model_name}")
        print("-" * 60)
        print(f"Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 No Depression  |  Depression")
        print(f"Actual  No Dep   TN = {tn:3d}       |  FP = {fp:3d}")
        print(f"        Dep      FN = {fn:3d}       |  TP = {tp:3d}")
        print(f"\nMetrics:")
        print(f"  True Negatives (TN):  {tn:3d} - Correctly predicted No Depression")
        print(f"  False Positives (FP): {fp:3d} - Incorrectly predicted Depression")
        print(f"  False Negatives (FN): {fn:3d} - Missed Depression cases")
        print(f"  True Positives (TP):  {tp:3d} - Correctly predicted Depression")
        print(f"\n  Total Samples: {total}")
        print(f"  Accuracy:      {data['accuracy']:.3f} ({data['accuracy']*100:.1f}%)")
        print(f"  Precision:     {data['precision']:.3f} ({data['precision']*100:.1f}%)")
        print(f"  Recall:        {data['recall']:.3f} ({data['recall']*100:.1f}%)")
        print(f"  Specificity:   {data['specificity']:.3f} ({data['specificity']*100:.1f}%)")
        print(f"  F1-Score:      {data['f1_score']:.3f}")
        print(f"  AUC-ROC:       {data['auc_roc']:.3f}")


if __name__ == "__main__":
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Load results
    results_file = project_root / 'data' / 'models' / 'results_20251112_131051.json'
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Please run train_models.py first to generate results.")
        exit(1)
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Create output directory
    output_dir = project_root / 'data' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRICES")
    print("="*80)
    
    # Create individual confusion matrices
    for model_key, data in results_data.items():
        model_name = data['model_name']
        cm = np.array(data['confusion_matrix'])
        accuracy = data['accuracy']
        
        save_path = output_dir / f'confusion_matrix_{model_key}.png'
        plot_confusion_matrix(cm, model_name, accuracy, save_path)
    
    # Create combined figure
    combined_path = output_dir / 'confusion_matrices_all_models.png'
    create_combined_confusion_matrices(results_data, combined_path)
    
    # Print detailed analysis
    print_confusion_matrix_details(results_data)
    
    print("\n" + "="*80)
    print("✓ ALL CONFUSION MATRICES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - confusion_matrix_logistic_regression.png")
    print("  - confusion_matrix_random_forest.png")
    print("  - confusion_matrix_gradient_boosting.png")
    print("  - confusion_matrix_deep_learning.png")
    print("  - confusion_matrices_all_models.png (Combined)")
