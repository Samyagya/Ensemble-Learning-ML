🧠 **Obesity Risk Prediction Using Ensemble Learning**

**🎯 Objective:**

Predict an individual’s risk of obesity using machine learning. With a structured dataset containing health, lifestyle, and demographic features the goal is to classify individuals into
different obesity risk categories by analysing patterns and building effective predictive models.

**📦 Dataset**

The data includes various features relevant to obesity prediction, such as eating habits, physical activity, and personal health indicators.

**✅ Baseline**

By using models like SVM, KNN and Decision Trees, we set a bechmark baseline to reach with ensemble learning models.


**🚀 Ensemble Techniques explored**

**1. Bagging (Bootstrap Aggregating)**
- Trains several models independently on different random subsets of the data.
- Final prediction: **majority vote (classification)** or **average (regression)**.
- 📌 *Example:* Random Forest
---
**2. Boosting**
- Models are trained **sequentially**, each focusing on the previous model’s errors.
- Final prediction: **weighted combination** of all models.
- 📌 *Popular Algorithms:* AdaBoost, Gradient Boosting, XGBoost, LightGBM
---
**3. Stacking (Stacked Generalization)**
- Combines multiple base models using a **meta-learner** trained on their outputs.
- Base models learn from the original data; the meta-learner learns from their predictions.
---
**4. Blending**
- A simpler form of stacking using a **hold-out validation set** instead of cross-validation.
---
**5. Voting & Weighted Voting**
- Combines predictions from multiple models:
    - **Hard Voting:** Majority class wins
    - **Soft Voting:** Average predicted probabilities
    - **Weighted Voting:** Give more influence to better-performing models
---
**6. Model Diversity**
Improves ensemble performance by using a diverse set of base models:
- Different algorithms
- Different hyperparameters
- Different subsets of features
- 

**📊 Evaluation Criteria**
- Spliting the dataset into **training and testing sets** (e.g., 80/20 split or use cross-validation).
- Using key metrics to evaluate performance:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-score**
  
Finally these results are clearly depicted using graphs and tables. Plots are made using matplotlib.py.
