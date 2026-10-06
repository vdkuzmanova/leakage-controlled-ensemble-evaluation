# Cross-dataset summary


## Pima Indians Diabetes Dataset
- n=768 (768 distinct records), 10x10 nested CV
- Best single: RandomForest (0.835 ± 0.046); best ensemble: Voting(RF+XGB) (0.839 ± 0.048)
- Ensemble − single: +0.0034, corrected p=0.523, P(equivalent, ROPE ±0.01)=0.89
- SHAP (RandomForest): Glucose, BMI, Age, Insulin, Pregnancies
- SHAP agreement (Kendall τ): LogisticRegression vs RandomForest: 0.57, LogisticRegression vs XGBoost: 0.64, RandomForest vs XGBoost: 0.79

## Early Stage Diabetes Risk Prediction Dataset
- n=520 (251 distinct records), 10x10 nested CV
- Best single: SVM (0.978 ± 0.040); best ensemble: Voting(RF+XGB) (0.974 ± 0.035)
- Ensemble − single: -0.0044, corrected p=0.730, P(equivalent, ROPE ±0.01)=0.54
- SHAP (RandomForest): Polyuria, Polydipsia, Gender, sudden weight loss, partial paresis
- SHAP agreement (Kendall τ): LogisticRegression vs RandomForest: 0.78, LogisticRegression vs XGBoost: 0.58, RandomForest vs XGBoost: 0.60