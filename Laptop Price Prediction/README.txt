AIC354 Machine Learning Fundamentals - Assignment 3
Laptop Price Prediction (10-Fold Cross Validation)

Student      : S.M.MATEEN UD DIN
Reg. No.     : FA24-BSE-152
Live web app : (add your Streamlit link here after deployment)

Files
-----
laptop_price_kfold_10fold.ipynb  Code (Jupyter / Google Colab notebook, with outputs)
laptop_price.csv                 Dataset (1,303 laptops, price in EUR)
laptop_price_model.pkl           Trained model (Hist Gradient Boosting pipeline)
app.py                           Streamlit web app (deployment code)
requirements.txt                 Libraries for the web app (same versions as training)

Algorithms compared (10-fold CV, mean RMSE in EUR, lower is better):
Hist Gradient Boosting 274 | Gradient Boosting 289 | Support Vector Regression 310 |
K-Nearest Neighbors 340 | Lasso Regression 420   (baseline: always the average price = 697)
