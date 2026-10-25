AIC354 Machine Learning Fundamentals - Assignment 3
Laptop Price Prediction (10-Fold Cross Validation)

Student      : S.M.MATEEN UD DIN
Reg. No.     : FA24-BSE-152
Live web app : (add your Streamlit link here after deployment)

Files
-----
laptop_price_kfold_10fold.ipynb  Code (Jupyter / Google Colab notebook, with outputs)
laptop_price.csv                 Dataset (1,303 laptops, original price column in EUR)
laptop_price_model.pkl           Trained model (Hist Gradient Boosting pipeline)
app.py                           Streamlit web app (deployment code)
requirements.txt                 Libraries for the web app (same versions as training)

Algorithms compared (10-fold CV, mean RMSE in PKR at 1 EUR = 316 PKR, lower is better):
Hist Gradient Boosting 86,600 | Gradient Boosting 91,300 | Support Vector Regression 98,000 |
K-Nearest Neighbors 107,400 | Lasso Regression 132,800   (baseline: always the average price = 220,300)

Note: the web app (app.py) shows all prices in Pakistani rupees (PKR). The model itself is trained on
the euro prices in the dataset, and app.py converts them with EUR_TO_PKR = 316 (edit it at the top of app.py).
