# Machine Learning Data Concepts

## 1. Data Types and Variables

Dataset is the foundation of machine learning and statistical modeling.

### 1.1 Numerical (Quantitative)

- **Continuous:** Real numbers on an unbroken measurement scale.
  - _Examples:_ Temperature ($23.5^\circ\text{C}$), Annual Salary ($\$75,400.50$), Distance ($12.8\text{ km}$).

- **Discrete (Counts):** Countable whole numbers with distinct integer gaps.
  - _Examples:_ Number of children ($2$), Daily website visits ($1,450$), Number of hospital beds ($120$).

### 1.2 Categorical (Qualitative)

- **Nominal:** Distinct labels or groups with no inherent mathematical order or ranking.
  - _Examples:_ Blood Type (A, B, AB, O), City of Residence (Tokyo, London, Chicago), Device Type (Mobile, Desktop, Tablet).

- **Ordinal:** Categories with a defined, natural hierarchy or sequence, but non-measurable intervals.
  - _Examples:_ Education Level (High School, Bachelor's, PhD), Customer Rating (1-Star, 3-Star, 5-Star), Shirt Size (S, M, L, XL).

- **Binary (Dichotomous):** Categorical variables restricted to exactly two mutually exclusive states.
  - _Examples:_ Subscription Status (Active / Inactive), Fraud Detection (0 / 1), Loan Approved (Yes / No).

### 1.3 Temporal and Datetime

- **Timestamp / Point-in-Time:** Specific chronological moments.
  - _Examples:_ Transaction Timestamp (`2026-08-26 13:15:00`), Date of Birth (`1995-04-12`), System Log Time (`08:00:12.450`).

- **Interval / Duration:** Length of elapsed time between two events.
  - _Examples:_ User Session Length ($4\text{ mins } 12\text{ secs}$), Delivery Time ($3\text{ days}$), Machine Uptime ($48.5\text{ hours}$).

### 1.4 Text and Unstructured

- **Freeform Text:** Open-ended natural language strings.
  - _Examples:_ Product Review ("Battery lasts all day"), User Bio ("Software engineer and runner"), Support Ticket Description ("Cannot connect to VPN").

- **Structured Strings / Codes:** Text following a strict syntactic rule or alphanumeric format.
  - _Examples:_ Postal Code (`94016`), SKU Code (`PROD-882-XL`), IP Address (`192.168.1.1`).

### 1.5 Identifiers and Metadata

- **Unique Key:** Identifiers used solely for indexing and tracking rows or entities.
  - _Examples:_ User ID (`USR_9021`), Transaction Hash (`0x7f...`), Social Security Number / National ID.

## 2. Key Features and Characteristics to Check

### 2.1 Continuous Numerical

Distribution shape: it tells where most of your data points live and whether you have weird outliers. It helps choose the right ML algorithm and validate your model's assumptions.
Primary distribution shape: 1. Normal (symmetric/bell-shaped) - mean (average), median (middle number) and mode (most common number) are all exactly the same number right in the center. E.g., human heights. 2. Right skewed (positively skewed) - looks like a hill on the left that slides down into a long, flat tail on the right. Mean is pulled to the right and is larger than the median. E.g., household income. How to fix - Log Transformation to squish the long tail and make the distribution look more normal before training. [mean -> RIGHT > Median] 3. Left skewed (negatively skewed) - a long, flat tail starting on the left that climbs up into a steep hill on the right. Mean is pulled to the left and smaller than the median. E.g., age of retirement. [LEFT <- mean < Median] 4. Bimodal (two peaks) - two separate hills with a valley in between them. E.g., restaurant traffic. Bimodal shape usually means you're accidentally mixing two completely different groups together in one column. Solution - separate features. \* Modality - number of peaks (humps) in data distribution. unimodal(one peak), bimodal(two peaks), multimodal(three or more peaks) and uniform(no peak)
**NOTE**: When validating assumptions, we use two metrics to measure these shapes (skewness, kurtosis):
Skewness: - measures how off-center or asymmetrical the shape is
: 0 is perfect symmetric
: +ve right skewed;
: -ve left skewed.
Kurtosis: - measure how thick the tails are and how sharp the center peak is.
: 👆(high) very sharp, tall peak with heavy tails (lots of extreme outliers)
: 👇(low) graph is flat and spread out like plateau

- **Scale and Range:** Check the minimum, maximum, and spread of values.
- **Outliers:** Identify observations that are unusually far from the rest of the data.
- **Missingness:** Measure how many values are missing and whether missingness is concentrated in particular groups.

### 2.2 Discrete Numerical (Count)

- Sparsity & Zero Inflation: Sparsity (empty space) - most cells are naturally empty or zero in a data table. How to fix: using a sparse matrix (compress data by only recording the non-zero numbers), dimensionality reduction (truncated SVD or embeddings). Zero inflation (too many zeros): a single continuous numeric target column contains way more zeros than a standard statistical distribution can explain. Solution: two-stage model, hurdle model.
- Upper bounds
- Dispersion - how spread out or scattered data points are around their central value (mean or average). The mean tells where the center of data lives, and dispersion tells how reliable that center actually is.
  How Measure dispersion - Range - Variance (Average squared distance): how far the data point from mean. - Standard Deviation : square root of variance.
  E.g., high and low dispersion - Low dispersion (high consistency): Archer A hits the inner bullseye 3 times, and the other 7 arrows land tightly packed right next to it. Their shots have a small standard deviation. The average position is dead-center, and you can highly trust them to hit near the center on the next shot.High Dispersion (High Volatility): Archer B hits the top-left edge of the target 5 times, and the bottom-right edge 5 times. Technically, if you calculate the mathematical average of their shots, it lands perfectly dead-center in the bullseye! But their standard deviation is massive. Their data is highly scattered, and you cannot trust where the next shot will go.

### 2.3 Nominal Categorical (Unordered)

- Cardinality - number of unique categories (distinct values) present in that column. Low cardinality can be handled using one-hot encoding. High cardinality causes problems and can be handled using techniques like frequency count, target encoding (out-of-fold), grouping rare values, or categorical embeddings with CatBoost's native encoding.
- Category Frequency - measures how often different text labels, classes, or distinct groups appear in a dataset.
- Train/Test Drift - Also called data drift or covariate shift, occurs when the statistical properties of the data you use to train your model change completely by the time it reaches the test phase or live production.
- Three types of drift
  - Data drift (Covariate shift) - Distribution of features changes but the underlying rule to predict the target remains same.
  - Concept Drift - distribution of inputs stays the same, but the real-world relationship between X and Y changes.
  - Label Drift (Prior Probability Shift) - Distribution of your target variable (Y) shifts independently.
- How to detect train/test drift using statistical validation tests
  - For numerical columns (the Kolmogorov-Smirnov (KS) test): compare the curve of training data against the new test data; if the distance between the curves is too large (low p-value), drift is detected.
  - For Categorical Columns (Chi-Square Test): Compares the frequencies of your text categories over time to see if the proportion of groups has broken away from the baseline
  - The Master Metric (Population Stability Index (PSI)) - calculates a single stability score. A score above 0.2 indicates your test distribution is no longer stable compared to training
  - Adversarial Validation Trick: combine your training and testing data into one file, erase the actual targets, and create a fake label: Is_This_Training_Data? (1 for Yes, 0 for No). You then train a model to try and tell them apart. If the model easily succeeds (High Accuracy), it proves your train and test sets look so different that drift has occurred
- How to fix it
  - Continuous Retraining Pipelines: Set up automated triggers so that when a drift metric (like PSI) breaches a threshold, the model automatically pulls the latest month of data and retrains its weights.
  - Feature Dropping: If a single input feature drifts wildly due to an external change (like a broken tracking metric) without helping the model much, it is safest to delete that feature entirely.
  - Sample Weighting: Give higher mathematical importance (weight) to newer data points and slowly phase out ancient data from years ago

## 3. Assumption Validation

**Assumption validation:** Assumption validation is the process of checking whether the statistical or structural conditions required by a specific algorithm are actually met by the data.

### Why This Is Required

- **Model reliability:** Violating a core assumption leads to false conclusions, high error rates, or biased predictions.
- **Algorithm suitability:** It helps determine if the chosen model is appropriate or if you should switch to a non-parametric model.

Here is the complete master list of assumption validations required in machine learning, organized by the specific category of the ML problem.

### 3.1 Regression Problems (Continuous Numbers)

#### 📈 Overview

Linear models are the strictest when it comes to assumptions. If you are using Linear Regression, you must validate these five core assumptions. If you use tree-based models (like Random Forests), these do not apply.

- Linearity: The relationship between the independent variables (X) and the dependent variable (Y) must be a straight line.
- How to validate: Look at a Scatter Plot or a Residual Plot. The points should form a random cloud around zero, not a curve.
- Homoscedasticity (Equal Variance): The spread of your model's errors (residuals) must be constant across all levels of your data.
- How to validate: Check the Residual vs. Fitted plot. If the plot fans out like a megaphone, this assumption is broken.
- Normality of Residuals: The errors made by the model must follow a perfect, symmetric bell-curve (normal distribution).
- How to validate: Use a Q-Q Plot (Quantile-Quantile) or a Shapiro-Wilk test. The points on a Q-Q plot should follow a straight diagonal line.
- Independence of Errors (No Autocorrelation): The error of one data point should not predict or affect the error of the next data point.
- How to validate: Run a Durbin-Watson test. A score near 2.0 means independence.
- No Multicollinearity: Your input features must not be highly correlated with each other (e.g., having both "Height in Inches" and "Height in Centimeters" as inputs).
- How to validate: Calculate the VIF (Variance Inflation Factor) for each feature. A VIF score above 5 or 10 means trouble.

### 3.2 Classification Problems (Categories)

#### 🏷️ Overview

Classification models have fewer spatial assumptions than regression, but they have critical assumptions regarding data balance, probability, and boundaries.

- Multiclass/Binary Balance (No Class Imbalance): Assumes the classes are relatively balanced so the model doesn't just guess the majority class to get a fake high score.
- How to validate: Check the frequency count of your labels. If imbalanced, validate using StratifiedKFold cross-validation instead of regular K-Fold.
- Independence of Features (Naive Bayes): If using a Naive Bayes classifier, it strictly assumes that every single input feature is completely independent of the others.
- How to validate: Use a Correlation Matrix. High correlation breaks this algorithm.
- Linear Separability (Logistic Regression / Linear SVM): Assumes a straight line or flat plane can cleanly cut through your data to separate the classes.
- How to validate: Use a Pair Plot to visually check if classes cluster separately. If they don't, you must switch to non-linear algorithms (like Kernel SVM or Decision Trees).

### 3.3 Time-Series Forecasting Problems (Sequential Data)

#### 🕰️ Overview

Because time series data naturally violates the rule that "data points must be independent," it requires a completely unique set of validations.

- Stationarity: Assumes that the data's statistical properties—like the mean, variance, and autocorrelation—are constant over time (no shifting trends or widening swings).
- How to validate: Run an ADF (Augmented Dickey-Fuller) Test or a KPSS Test. If p-value < 0.05 on ADF, it is stationary.
- Autocorrelation (Lag Dependencies): Assumes that past values have a measurable, predictable relationship with future values.
- How to validate: Look at ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots to find the exact time lag patterns.
- No Data Leakage in Validation: Assumes your validation strategy respects time order. You cannot use future data to predict the past.
- How to validate: Ensure your cross-validation uses TimeSeriesSplit (rolling window evaluation) instead of random shuffling.

#### Ordinal Categorical Data (Ordered)

- **Order Integrity:** Ensuring that the sequence or chronological arrangement of your data points is completely preserved during data cleaning, transformation, and model evaluation. This is the number one assumption validation requirement for time-series forecasting and sequence models.
- In sequential data, the position is the signal. Violations of order integrity typically happen in two ways:
  - The Shuffling Trap (Broken Validation):
  - Sorting Loss during Merges & Joins
- How to protect order integrity:
  - Chronological Splitting (the time barrier): Use a strict time boundary line to carve out your data, or use specialized tools like Scikit-Learn's TimeSeriesSplit.
  - Monotonic Index Checking: Mathematically verify that your timestamps or sequence steps move in exactly one forward direction without gaps or backward jumps.
  - Avoiding in-place data shuffling:
  - **Data Leakage:** When information from outside the training dataset is accidentally used to train a machine learning model.
  - Types of data leakage: Target leakage (including the answers), train-test contamination (mixing data slices).
- **Interval Spacing:** Gaps or distance between sequential data points

### 3.4 Clustering Problems (Unsupervised Grouping)

#### 👥 Overview

Distance-based clustering algorithms (like K-Means) fail if the geometric shape of your data doesn't match their mathematical design.

- Spherical Clusters (K-Means): Assumes that your hidden groups are shaped like round, even balls (spheres) and are roughly the same size.
- How to validate: Use a Scatter Plot. If your clusters look like long snakes, crescent moons, or varied sizes, K-Means will fail (use DBSCAN instead).
- Feature Scaling Sensitivity: Assumes all features are on the exact same numerical scale because it calculates physical distance to group items.
- How to validate: Compare the minimum and maximum ranges of your columns. If one column ranges from 0–1 and another from 0–1,000,000, you must apply StandardScaler or MinMaxScaler.

### 3.5 General Machine Learning Assumptions (Applies to ALL)

#### 🚨 Overview

No matter what type of ML problem you are solving, these universal assumptions must be verified.

- I.I.D. (Independent and Identically Distributed): Assumes every row of data is collected independently of the others, and all rows are drawn from the same real-world environment.
- Representative Data: Assumes your training data looks exactly like the data the model will see out in the real world in the future.
- Low Modality in Target Errors: Assumes your final error distribution is unimodal (one peak). A bimodal or multimodal error shape means your model is missing a critical hidden variable or sub-group.

## 4. Recommended Models by Data Type

| Data Type  | Primary Go-To Models        | Secondary Diverse Models                      |
| ---------- | --------------------------- | --------------------------------------------- |
| Tabular    | LightGBM, XGBoost, CatBoost | TabNet, ResNet/MLP for tabular, Random Forest |
| Text / NLP | DeBERTa-v3, RoBERTa         | TF-IDF + Ridge/GBDT, FastText                 |
| Vision     | ConvNeXt, EfficientNet-v2   | Swin Transformer, EVA-02                      |

## To Be Discussed

- Data leakage
- Detecting leaks and shifts
- Adversarial train/test drift
- Power transforms (e.g., Yeo-Johnson)
