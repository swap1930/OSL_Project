#!/usr/bin/env python3
"""
Simple CI test script for OSL-SLA project
Tests model training and basic functionality
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# Add src to path
sys.path.insert(0, 'src')

def main():
    print("Starting OSL-SLA CI Test Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Train model
        print("Step 1: Training model...")
        from main import main as train_main
        train_main()
        print("Model training completed")
        
        # Step 2: Test model with sample data
        print("\n Step 2: Testing model predictions...")
        from src.predict import predict
        
        # Test cases with different scenarios
        test_cases = [
            # [Air temp, Process temp, RPM, Torque, Tool wear]
            [300.0, 310.0, 1500.0, 40.0, 50.0],   # Normal operation
            [320.0, 330.0, 2000.0, 80.0, 200.0],  # High stress
            [280.0, 290.0, 1000.0, 20.0, 10.0],   # Low stress
        ]
        
        results = []
        for i, test_data in enumerate(test_cases, 1):
            try:
                prediction, probability = predict(test_data)
                results.append((test_data, prediction, probability))
                status = "FAILURE" if prediction == 1 else "NORMAL"
                print(f"  Test {i}: {status} (Prob: {probability:.3f})")
            except Exception as e:
                print(f"  Test {i}:  ERROR - {e}")
                return False
        
        print("Model predictions completed")
        
        # Step 3: Load test data for metrics
        print("\n Step 3: Evaluating model performance...")
        
        # Create synthetic test data for evaluation
        np.random.seed(42)
        n_test_samples = 100
        
        test_data = {
            "UDI": range(1, n_test_samples + 1),
            "Product ID": [f"T{i:05d}" for i in range(1, n_test_samples + 1)],
            "Type": np.random.choice(['M', 'L', 'H'], n_test_samples),
            "Air temperature [K]": np.random.normal(300, 10, n_test_samples),
            "Process temperature [K]": np.random.normal(310, 10, n_test_samples),
            "Rotational speed [rpm]": np.random.normal(1500, 300, n_test_samples),
            "Torque [Nm]": np.random.normal(40, 15, n_test_samples),
            "Tool wear [min]": np.random.uniform(0, 250, n_test_samples),
            "Machine failure": np.random.binomial(1, 0.1, n_test_samples),  # ~10% failure rate
        }
        
        test_df = pd.DataFrame(test_data)
        
        # Preprocess test data
        from src.preprocessing import preprocess
        X_test, y_test = preprocess(test_df)
        
        # Make predictions on test data
        predictions = []
        probabilities = []
        for i in range(len(X_test)):
            pred, prob = predict(X_test[i].tolist())
            predictions.append(pred)
            probabilities.append(prob)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        print(f"Model Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  Test Samples: {n_test_samples}")
        print(f"  Failure Rate: {np.mean(y_test):.3f}")
        
        # Step 4: Check model files
        print("\n Step 4: Verifying model files...")
        model_files = [
            "model/model.pkl",
            "model/scaler.pkl"
        ]
        
        all_files_exist = True
        for file_path in model_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"   {file_path} ({size} bytes)")
            else:
                print(f"   {file_path} - MISSING")
                all_files_exist = False
        
        if not all_files_exist:
            return False
        
        print("All model files verified")
        
        # Final summary
        print("\n CI Pipeline Summary:")
        print("=" * 50)
        print("Model training: SUCCESS")
        print("Model predictions: SUCCESS") 
        print("Performance evaluation: SUCCESS")
        print("Model files: VERIFIED")
        print(f"Final Metrics - Acc: {accuracy:.3f}, Prec: {precision:.3f}, F1: {f1:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n CI Pipeline Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
