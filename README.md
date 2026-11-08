# Credit Risk Classification

A machine learning project that predicts whether a mortgage applicant is likely to default on their loan. Built for my Introduction to Deep Learning course at Pace University (Spring 2026).

## The Problem

Banks lose money when they approve loans that end up defaulting. The goal here was to train a model that sorts applicants into high-risk and low-risk, so risky loans can be flagged before they're approved.

## Dataset

148,670 real mortgage records with 31 features, including loan amount, income, credit score, property value, and loan-to-value ratio. The target column, `Status`, marks whether the loan defaulted.

The dataset isn't included in this repo because of its size. You can download it from Kaggle here: [https://www.kaggle.com/datasets/yasserh/loan-default-dataset]

## What I Did

1. Cleaned the data: removed duplicates, filled in missing values, encoded text categories as numbers, and scaled numeric features
2. Split the data 80/20 into training and test sets
3. Trained two models: Logistic Regression as a baseline and a Random Forest with 100 trees
4. Evaluated both with accuracy, precision, recall, and confusion matrices

## Catching Target Leakage

My first Random Forest scored 99.99% accuracy, which was too good to be true. I traced it to three columns: `rate_of_interest`, `Interest_rate_spread`, and `Upfront_charges`. Lenders set these based on how risky they already think a borrower is, so the model was basically reading the answer instead of learning from the applicant's actual data. After removing them, I got the honest results below.

## Results

| Model | Accuracy | Defaults Caught (Recall) | Missed Defaults |
|---|---|---|---|
| Logistic Regression | 77.79% | 19% | 5,886 |
| Random Forest | 89.13% | 60% | 2,925 |

**Random Forest was the clear winner.** It caught three times as many defaulters, and when it flagged someone as high-risk it was right 93% of the time. In credit risk, a missed default is the most expensive mistake, so recall on defaults mattered more than overall accuracy.

## What I Learned

- **Accuracy can hide a bad model.** Logistic Regression scored 78% accuracy but missed 81% of actual defaults. The confusion matrix showed what the accuracy number didn't.
- **You need to understand your data.** I only caught the leakage because I went through the data dictionary and thought about what each column meant in real life.

## Next Steps

- Handle the class imbalance (there are far more repaid loans than defaults) with class weights or SMOTE
- Try XGBoost
- Lower the decision threshold to catch more defaults, even if that means more false alarms

## How to Run

1. Download the dataset from Kaggle and put the CSV in the same folder as the notebook
2. Install the libraries: `pip install pandas numpy scikit-learn matplotlib seaborn` (Anaconda already includes them)
3. Open `credit_risk.ipynb` in Jupyter or VS Code and run all cells

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter
