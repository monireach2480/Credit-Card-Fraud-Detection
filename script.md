

**Video Script**
**0:00 - Introduction**

Hello, we are **Len Monireach, KHLIM Vinchhieng, and PHAT Vicheymongkol** from Group 8, and this is a demonstration of our final machine learning project: Credit Card Fraud Detection.

The goal of this project is to detect fraudulent credit card transactions using machine learning. The main challenge is that the dataset is highly imbalanced. Most transactions are legitimate, and only a very small percentage are fraud, so accuracy alone is not a useful metric.

Because of that, we focused more on precision, recall, F1-score, ROC-AUC, and especially PR-AUC.


**0:35 - Project Structure**

Here is the project folder.

The `data` folder contains the raw and processed datasets.  
The `notebooks` folder contains the full machine learning workflow, from data exploration to preprocessing, baseline models, XGBoost tuning, evaluation, explainability, and final deployment export.

The `src` folder contains reusable Python helper files for preprocessing, training, evaluation, SHAP explainability, and prediction utilities.

The `api` folder contains the FastAPI backend, and the `app` folder contains the Streamlit frontend. The `models` folder stores the trained models, and the `reports` folder stores result tables and figures.

**1:15 - Problem With Original Notebook 4**

The original XGBoost tuning in Notebook 4 had one major issue. It used `scoring="recall"` during hyperparameter tuning.

This caused the model to focus too much on catching fraud cases, but precision became very low. In other words, the model was flagging too many legitimate transactions as fraud.

For fraud detection, recall is important, but precision also matters because too many false alarms can create problems for customers and banks.

So the improvement was not just to get a higher score, but to make the model more balanced and practical.

**1:55 - Notebook 6 Improvements**

Now I will show the new improvement notebook, which is currently saved as `notebooks/improvement.py`.

This notebook adds four major improvements.

First, we added SHAP feature analysis. SHAP was used to measure which features contributed most to the model predictions. The most important features included `V14`, `V4`, `V12`, `V11`, `V10`, `Time`, `V3`, and `V8`.

The project tested both the full feature set and the SHAP-selected feature set. The full feature set performed better, which is still an important result. It shows that even features with smaller individual importance can still help the model when used together.

Second, we fixed the XGBoost tuning metric. Instead of tuning for recall only, the new notebook uses `average_precision`, which represents PR-AUC. This is more appropriate for imbalanced classification.

Third, we added LightGBM as a new model. This gave us another advanced tree-based model to compare against XGBoost.

Fourth, we replaced the manual threshold testing with an optimal threshold search using the precision-recall curve. Instead of only trying thresholds like 0.3, 0.4, 0.5, 0.6, and 0.7, the new method checks all possible thresholds from the precision-recall curve and chooses the one with the best F1-score.

**3:10 - Results**

Now I will show the final model comparison table.

The baseline Logistic Regression model had a precision of about 0.827, recall of 0.633, F1-score of 0.717, and PR-AUC of 0.741.

Random Forest performed better, with precision around 0.941 and F1-score around 0.874.

The old best XGBoost model from Notebook 4 had precision around 0.882, recall around 0.837, F1-score around 0.859, and PR-AUC around 0.880.

The new LightGBM model was tested, but it did not outperform XGBoost on this dataset.

The final best model is XGBoost Tuned plus Optimal Threshold. It achieved precision around 0.942, recall around 0.827, F1-score around 0.880, ROC-AUC around 0.981, and PR-AUC around 0.881.

The improvement is modest, but meaningful. Precision improved, F1-score improved, and ROC-AUC improved, while recall only dropped slightly.

This makes the final model more practical because it reduces false fraud alerts while still detecting most fraud cases.

**4:20 - API Demo**

Next, I will demonstrate the backend API.

Here is the FastAPI documentation page. The API has a health endpoint and a prediction endpoint.

The backend loads the improved XGBoost model from `models/improved/xgb_tuned_improved.pkl`. It also loads the optimized threshold from `models/improved/xgb_best_threshold.pkl`.

I will first run the health endpoint.

This confirms that the API is running, the model is loaded, and the system is ready to make predictions.

Now I will test the prediction endpoint with a sample transaction.

The API accepts transaction features including `Time`, `Amount`, and anonymized PCA features from `V1` to `V28`.

The output returns the prediction, fraud probability, decision threshold, and label.

**5:10 - Streamlit Demo**

Now I will show the Streamlit frontend.

This interface allows a user to enter transaction values manually or paste a JSON transaction.

The Streamlit app sends the transaction to the FastAPI backend and displays the result.

I will use the sample transaction and click predict.

The app shows whether the transaction is classified as fraud or legitimate, along with the fraud probability and the threshold used by the final model.

This demonstrates that the machine learning model is not only trained and evaluated, but also deployed into a usable prediction system.

**5:55 - Conclusion**

To summarize, this project built a complete fraud detection pipeline.

We handled class imbalance, compared baseline models, trained XGBoost, added SHAP explainability, tested LightGBM, improved the tuning method, optimized the classification threshold, and deployed the final model with FastAPI and Streamlit.

The final model is XGBoost Tuned plus Optimal Threshold. It provides the best overall balance between precision, recall, F1-score, and PR-AUC for this project.

Thank you for watching our project demonstration.