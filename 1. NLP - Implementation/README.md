👉 Scikit-Learn makes building ML models easy, fast, and reliable without writing everything from scratch.

Alternate for Scikit-Learn: We can also use Gensim, TensorFlow, PyTorch.

## 📚 ML / DL Tools: Library vs Framework

| Tool / Package | Type | Category | Best Used For |
|---------------|------|----------|----------------|
| scikit-learn (sklearn) | Library | Classical ML | Classification, regression, clustering, preprocessing, BoW/TF-IDF |
| Statsmodels | Library | Statistical ML | Statistical analysis, regression with p-values, time series |
| PyTorch | Library (DL Framework) | Deep Learning | Custom neural nets, CNNs, RNNs, Transformers, research |
| TensorFlow | Framework | Deep Learning | End-to-end DL pipelines, production deployment |
| Keras | High-level API / Framework | Deep Learning | Easy neural network building on TensorFlow |
| XGBoost | Library | Boosting ML | High-performance models for tabular data |
| LightGBM | Library | Boosting ML | Fast gradient boosting for large datasets |
| CatBoost | Library | Boosting ML | Boosting with categorical features |
| Gensim | Library | NLP | Word2Vec, Doc2Vec, topic modeling (LDA) |
| NumPy | Library | Numerical Computing | Arrays, linear algebra, core math ops |
| Pandas | Library | Data Processing | DataFrames, cleaning, analysis |

Finally Implemeting the E-Mail Classification Problem for better understanding of the basic concepts of the NLP.

---

---

## **Spam Classification Pipeline**

```
Raw SMS text
   ↓
Text Cleaning & Preprocessing
   ↓
Corpus (cleaned sentences)
   ↓
Vectorization (BoW / TF-IDF)
   ↓
Train–Test Split
   ↓
Model Training (Naive Bayes)
   ↓
Prediction
   ↓
Evaluation (Accuracy + Classification Report)
```

---

## 🌲 Why `RandomForestClassifier` Does Not Have `fit_transform()`

In scikit-learn, different components play different roles in a machine learning pipeline.  
Some are used to **transform data**, while others are used to **learn from data and make predictions**.

Understanding this explains why `RandomForestClassifier` does not implement `fit_transform()`.

---

## **Using Transformer vs Estimator Pipeline**

```
Raw Data
   ↓
Transformer (e.g., TF-IDF, Scaler)
   ↓
Features
   ↓
Estimator (e.g., Naive Bayes, RandomForest)
   ↓
Predictions
```

---

## Transformers vs Estimators (Models)

scikit-learn broadly categorizes tools into:

### **1. Transformers — Change the data**

Transformers are used to convert raw data into a new feature representation.  
They learn parameters from the data and then apply a transformation.

They provide:
- **fit()** → learn from data  
- **transform()** → transform data  
- **fit_transform()** → fit + transform in one step  

**Examples:**
- CountVectorizer (text → word counts)  
- TfidfVectorizer (text → TF-IDF features)  
- StandardScaler (scale features)  
- PCA (dimensionality reduction)  

```python
X = TfidfVectorizer().fit_transform(corpus)
```

### **2. Estimators / Models — Learn & Predict**

Estimators (also called models) are used to learn a mapping from input features **X** to output labels **y**.  
Their role is to **make predictions**, not to change the data representation.

They provide:
- **fit()** → train the model  
- **predict()** → predict labels for new data  

They do **not** provide:
- **transform()**  
- **fit_transform()**

**Examples:**
- MultinomialNB  
- RandomForestClassifier  
- LogisticRegression  
- SVC  

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```