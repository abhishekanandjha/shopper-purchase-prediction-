# ML Assignment 2: Predicting Online Shopper Purchases

**Name:** Abhishek Anand Jha  
**BITS ID:** 2026AJ05089  
**Course:** Machine Learning, M.Tech (AIML)

**Live app:** <STREAMLIT_APP_LINK>

---

## a. Problem statement

An e-commerce site gets thousands of browsing sessions a day, but only a small share of them end in a purchase.
The aim here is to predict, from the way a visitor browses (pages visited, time spent, bounce and exit rates and so on),
whether the session will end with the visitor buying something.

This is a **binary classification** problem: `Revenue = 1` (purchase) or `Revenue = 0` (no purchase).
Five classifiers are trained on the same data and compared on six metrics, and the trained models are served
through a Streamlit app where a test CSV can be uploaded and evaluated.

## b. Dataset description

**Source:** UCI Machine Learning Repository, *Online Shoppers Purchasing Intention Dataset* (ID 468)
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

- **Instances:** 12,330 sessions, each from a different user, collected over one year (12,205 after dropping 125 exact duplicate rows)
- **Features:** 17 input features plus the target
- **Target:** `Revenue` (True/False)
- **Class balance:** 10,422 no purchase vs 1,908 purchase, so only about **15.5%** positives. The data is clearly imbalanced, which is why accuracy alone is not a good measure here and F1 / MCC are given more weight.

| Feature | Type | Meaning |
|---|---|---|
| Administrative, Informational, ProductRelated | numeric | number of pages of each type visited in the session |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | numeric | time spent (seconds) on each page type |
| BounceRates | numeric | average bounce rate of the pages visited |
| ExitRates | numeric | average exit rate of the pages visited |
| PageValues | numeric | average Google Analytics "page value" of the pages visited |
| SpecialDay | numeric | closeness of the visit to a special day such as Valentine's Day (0 to 1) |
| Month | categorical | month of the visit |
| OperatingSystems, Browser, Region, TrafficType | categorical (integer codes) | visitor's technical and traffic details |
| VisitorType | categorical | Returning, New or Other |
| Weekend | boolean | whether the visit was on a weekend |

**Preprocessing**
- Dropped duplicate rows.
- Stratified 80/20 train/test split (`random_state=42`). The test part is saved as `test_data.csv`, which is what gets uploaded in the app.
- Numeric columns are standardised. `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region` and `TrafficType` are one-hot encoded (the last four are stored as numbers but they are only category codes). That gives 73 features after encoding.
- The preprocessing is stored inside each saved pipeline, so the app can take the raw CSV directly.

## c. GitHub repository link

https://github.com/abhishekanandjha/shopper-purchase-prediction-

```
ML_Assignment2/
├── app.py                  Streamlit app
├── train_models.py         trains and evaluates all 5 models, saves them to model/
├── preprocessing.py        column lists and cleaning shared by training and the app
├── requirements.txt
├── README.md
├── test_data.csv           20% held-out test split (raw format, includes Revenue)
├── data/
│   └── online_shoppers_intention.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics.csv         test metrics written by train_models.py
```

**Run locally**
```bash
pip install -r requirements.txt
python train_models.py      # retrains and overwrites model/ and test_data.csv
streamlit run app.py
```

## d. Models used

All five models use the same split and the same preprocessing. Settings used:

| Model | Settings |
|---|---|
| Logistic Regression | default (L2, C = 1), `max_iter=2000` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=20` |
| KNN | `n_neighbors=15` |
| Naive Bayes | Gaussian NB |
| Support Vector Machine | RBF kernel, C = 1, `gamma="scale"` |

The tree depth and K were picked with a quick 5-fold CV on the training set. For the tree, depths 3, 5 and 8 gave F1 around 0.64, while an unrestricted tree dropped to 0.56, so it clearly overfits without a limit. For KNN, K = 5 gave slightly better F1 (0.48) but lower AUC (0.78), while K = 15 gave AUC 0.83, so I went with 15 as a middle ground.

For SVM the AUC is computed from the decision function score instead of probabilities (`SVC(probability=True)` is deprecated in scikit-learn 1.9, and AUC only needs a ranking score anyway).

### Comparison table (test set, 2,441 sessions)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8873 | 0.8997 | 0.7560 | 0.4136 | 0.5347 | 0.5049 |
| Decision Tree | **0.9021** | **0.9263** | 0.7085 | 0.6361 | **0.6703** | **0.6143** |
| KNN | 0.8751 | 0.8459 | 0.7278 | 0.3220 | 0.4465 | 0.4289 |
| Naive Bayes | 0.2773 | 0.7511 | 0.1756 | **0.9791** | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8951 | 0.8727 | 0.7368 | 0.5131 | 0.6049 | 0.5586 |

Precision, recall and F1 are for the positive class (purchase). A model that predicts "no purchase" for everyone would already get about 84.4% accuracy, so accuracy on its own says very little here.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Ranks sessions quite well (AUC 0.90, second best) and has the highest precision (0.756), so when it says "will buy" it is usually right. But at the default 0.5 threshold it catches only 41% of actual buyers. The minority class pulls the probabilities down, and a linear boundary can't fully use the sharp jump in purchase rate once PageValues becomes non-zero. Lowering the threshold or using class weights would help recall. |
| Decision Tree | Best model on this data: highest accuracy, AUC, F1 and MCC, and the best recall among the sensible models (0.64). The data has strong threshold-type patterns. About 78% of sessions have PageValues = 0 and only about 4% of those buy, compared with 56% when PageValues > 0, so splits like "PageValues > x" suit a tree very well. PageValues alone accounts for about 78% of the tree's feature importance, followed by BounceRates and Month = Nov. Limiting the depth to 5 was important, because the unrestricted tree overfit badly. |
| KNN | Weakest of the reasonable models (F1 0.45, recall 0.32). Because only 15.6% of sessions are buyers, most neighbourhoods are dominated by non-buyers, so a buyer often gets outvoted. Distances are also less meaningful with 73 features, most of them one-hot columns. It also has to keep the whole training set in memory, which makes it the largest model file. |
| Naive Bayes | Looks very poor (accuracy 0.28) because it labels almost every session as a purchase. Recall is 0.98, but precision is only 0.18. The Gaussian assumption doesn't hold here: the duration and PageValues features are heavily right-skewed (skewness 5 to 7) with lots of exact zeros, and the one-hot 0/1 columns are being modelled as normal distributions. That produces extreme likelihoods. Its AUC of 0.75 shows it still ranks sessions somewhat, but the predicted probabilities are badly calibrated. |
| Support Vector Machine | Second best overall (F1 0.60, MCC 0.56). The RBF kernel picks up non-linear structure that logistic regression misses, giving better recall (0.51 vs 0.41) with similar precision. Its AUC (0.87) is a bit below logistic regression, and it is the slowest model to train and predict. It also doesn't give probabilities by default. |
| **Overall, winner for this dataset?** | **Decision Tree.** It has the best F1 (0.67) and MCC (0.61), the two metrics that matter most with this class imbalance, and also the best accuracy and AUC. It is easy to interpret: it basically says that sessions where the visitor reached valuable pages (high PageValues) with low bounce rates are the ones that convert. SVM comes second. Naive Bayes is not suitable for this data without first transforming the features. |

## Streamlit app

Features in the app:
1. **CSV upload** for test data (sidebar). There is also a button to download the sample `test_data.csv`, and a checkbox to use it directly.
2. **Model selection dropdown** for the five saved models.
3. **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) for the chosen model on the uploaded data.
4. **Confusion matrix** and **classification report**, plus a one-line summary of how many buyers were caught or missed.
5. A **Compare all models** tab that scores all five models on the same uploaded file, and a **Data preview** tab.

The app only loads the saved `.joblib` pipelines and calls `predict`. Nothing is retrained on the uploaded data.
The uploaded CSV must have the 17 feature columns and the `Revenue` column, in the same format as the original dataset.
