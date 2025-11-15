# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import requests
from io import BytesIO

url = "https://ars.els-cdn.com/content/image/1-s2.0-S0959652624001859-mmc1.xlsx"

# Add headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = BytesIO(response.content)
    df = pd.read_excel(data, header = 2)
    
else:
    print("Failed to fetch file, status code:", response.status_code)
df.head(10)

# %%
df.shape

# %%
df.columns

# %%
len(df.columns)

# %%
df['Solvent name'].unique()

# %%
df['Solvent name'].value_counts()

# %%
df2 = df.drop(['Access link', 'DOI'], axis= 'columns')
df2.head(5)
df2.shape

# %% [markdown]
# ## Data cleaning

# %%
df2.isnull().sum()
df2.shape

# %%
columns_to_keep_nulls = ['Conversion (%)', 'Phenol (wt%)', 'Catechol (wt%)', 'Cyclohexane (wt%)', 'Cyclohexanol (wt%)']  
columns_to_check = df2.columns.difference(columns_to_keep_nulls)

# Drop rows with nulls only in those columns
df3 = df2.dropna(subset=columns_to_check)

df3.isnull().sum
df3.shape

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# #### Add new feature(Heating rate (c/min))

# %%
df3['Heating_rate'] =  df3['Temperature (°C)']  / ( df3['Reaction time (h)'] * 60 )
df3.head(4)

# %%
df3.loc[2]

# %%
len(df3['Heating_rate'])

# %%
len(df3['Solvent quantity (mL)'])

# %%
solvent_type = df3.groupby('Solvent name')['Solvent name'].agg('count')
solvent_type.sort_values(ascending = False)

# %%
type(solvent_type)

# %%
solvent_type_less_than_3 = solvent_type[solvent_type <= 3]
len(solvent_type_less_than_3)

# %%
df3['Solvent name'] = df3['Solvent name'].apply(lambda x : "Other" if x in solvent_type_less_than_3 else x  )
solvent_type = df3.groupby('Solvent name')['Solvent name'].agg('count')
solvent_type.sort_values(ascending = False)

# %%
solvent_type_less_than_3

# %% [markdown]
# #### Outlier Removal Using domain knowledge

# %%
len(df3[df3['Heating_rate']> 4])

# %%
df4 = df3[~(df3['Heating_rate']> 4)]
len(df4)

# %% [markdown]
# ##### Outlier Removal Using Standard Deviation and Mean

# %%
df4['Temperature (°C)'].describe()


# %% [markdown]
# #### I removed outliers based on solvent type, because different solvents may naturally occurs ain different heating ranges.

# %%
def remove_temperature_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('Solvent name'):
        m = np.mean(subdf['Temperature (°C)'])
        st = np.std(subdf['Temperature (°C)'])
        reduced_df = subdf[(subdf['Temperature (°C)']>(m-st)) & (subdf['Temperature (°C)']<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df5 = remove_temperature_outliers(df4)
# df5.shape
df5.head(3)

# %%
import matplotlib.pyplot as plt
import pandas as pd
import math

# Select numeric columns
numeric_cols = df5.select_dtypes(include=['number']).columns
n = len(numeric_cols)

# Decide grid size (rows x cols)
cols = 4  # number of histograms per row
rows = math.ceil(n / cols)

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))  # adjust size
axes = axes.flatten()  # flatten to 1D for easy indexing

# Plot each histogram
for i, col in enumerate(numeric_cols):
    axes[i].hist(df5[col], bins=30, rwidth=0.8, color='skyblue', edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")

# Remove empty subplots (if any)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # adjust spacing
plt.show()


# %%
df6 = df5.drop(['Catalyst name' and 'Heating_rate' and 'Catalyst name'],axis='columns')
df6.head(3)

# %%
dum = pd.get_dummies(df6['Solvent name'])
df7 = pd.concat([df6.drop('Solvent name', axis= 'columns'),dum.drop('Other', axis= 'columns')], axis= 'columns')
df7.head(3)
# df11.to_csv('df11am.csv', index = False)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Feature columns
feature_columns = ['Crystallinity index (%)',
                   'BET surface area (m²/g)',
                   'Catalyst quantity (g)',
                   'Solvent quantity (mL)',
                   'Guaiacol (g)',
                   'H₂ pressure (MPa)',
                   'Temperature (°C)',
                   'Reaction time (h)']

# Correlation matrix
correlation_matrix = df7[feature_columns].corr()

# Smaller heatmap with Times New Roman bold
plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='BuPu', fmt=".2f", linewidths=0.5, cbar=False,
            annot_kws={"size":12, "fontname":"Times New Roman", "fontweight":"bold"})
plt.title('Feature Correlation Heatmap', fontsize=12, fontname="Times New Roman", fontweight="bold")
plt.xticks(fontsize=8, rotation=45, fontname="Times New Roman", fontweight="bold" )
plt.yticks(fontsize=8, rotation=0, fontname="Times New Roman", fontweight="bold")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Developing ML models

# %%
df8 = df7.dropna(subset=['Conversion (%)'])

# %%
# Columns to drop from the dataset entirely
drop_cols = ['Heating_rate']

# Your output (target) columns
y_cols = ['Conversion (%)', 'Phenol (wt%)', 'Catechol (wt%)',
          'Cyclohexane (wt%)', 'Cyclohexanol (wt%)']

# Create Y dataframe
Y = df8[y_cols]

# Create X dataframe: drop both the unwanted columns and target columns
X = df8.drop(columns=drop_cols + y_cols)

# Optional: check shapes
print("X shape:", X.shape)
print("Y shape:", Y.shape)


# %%
Y.info()

# %% [markdown]
# ### model (  conversion % )target

# %%
X.info()

# %%
from sklearn.model_selection import train_test_split

# Use Conversion (%) as the target
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y['Conversion (%)'],  # Correct column selection
    test_size=0.20, 
    random_state=10
)

# Optional: check shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# %%
# Import libraries
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Linear Regression
lr_clf = LinearRegression()
lr_clf.fit(X_train, Y_train)
print("Linear Regression R2:", lr_clf.score(X_test, Y_test))

# Decision Tree Regressor
dt_clf = DecisionTreeRegressor(random_state=45)
dt_clf.fit(X_train, Y_train)
print("Decision Tree R2:", dt_clf.score(X_test, Y_test))

# Random Forest Regressor
rf_clf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_clf.fit(X_train, Y_train)
print("Random Forest R2:", rf_clf.score(X_test, Y_test))

# XGBoost Regressor
xgb_clf = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=100,
    random_state=42
)
xgb_clf.fit(X_train, Y_train)
print("XGBoost R2:", xgb_clf.score(X_test, Y_test))

# MLP Regressor
mlp_clf = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_clf.fit(X_train, Y_train)
print("MLP Regressor R2:", mlp_clf.score(X_test, Y_test))


# %%
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import numpy as np

# Define cross-validation strategy
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

# List of models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=45),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=100,
        random_state=42
    ),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    # Fit model on full training set (optional, for later prediction)
    model.fit(X_train, Y_train)
    print(f"{name} R2 on test set: {model.score(X_test, Y_test):.4f}")
    
    # Cross-validation scores
    scores = cross_val_score(model, X, Y['Conversion (%)'], cv=cv, scoring='r2')
    print(f"{name} Cross-Validation R2 scores: {scores}")
    print(f"{name} Mean CV R2: {np.mean(scores):.4f}\n")


# %%
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import numpy as np

# Cross-validation strategy
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# Models dictionary including GBR
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=45),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting (GBR)": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "XGBoost": xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=100,
        random_state=42
    ),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Bagging Regressor": BaggingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, Y_train)
    print(f"{name} R2 on test set: {model.score(X_test, Y_test):.4f}")
    
    scores = cross_val_score(model, X, Y['Conversion (%)'], cv=cv)
    print(f"{name} Cross-Validation R2 scores: {scores}")
    print(f"{name} Mean CV R2: {np.mean(scores):.4f}\n")


# %%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Model names
model_names = []
mse_scores = []
r2_scores = []
cv_r2_scores = []

# Train and evaluate each model
for name, model in models.items():
    try:
        # Perform cross-validation
        scores = cross_val_score(model, X_train, Y_train, cv=cv, scoring='r2')
        
        # Train on full training set
        model.fit(X_train, Y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(Y_test, y_pred)
        r2 = model.score(X_test, Y_test)
        mean_cv_r2 = np.mean(scores)
        
        # Store results
        model_names.append(name)
        mse_scores.append(mse)
        r2_scores.append(r2)
        cv_r2_scores.append(mean_cv_r2)
        
    except Exception as e:
        pass

# Create a DataFrame for plotting
performance_df = pd.DataFrame({
    'Model': model_names,
    'MSE': mse_scores,
    'R_squared': r2_scores,
    'CV_R_squared': cv_r2_scores
})

# Sort by R-squared for better visualization
performance_df = performance_df.sort_values('R_squared', ascending=True)

# Plotting
plt.figure(figsize=(15, 6))

# Bar plot for MSE
plt.subplot(1, 3, 1)
sns.barplot(x='MSE', y='Model', data=performance_df, palette='viridis')
plt.title('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
plt.xlabel('MSE')

# Bar plot for R-squared
plt.subplot(1, 3, 2)
sns.barplot(x='R_squared', y='Model', data=performance_df, palette='viridis')
plt.title('R-squared Score (Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('R²')

# Bar plot for Cross-Validation R-squared
plt.subplot(1, 3, 3)
sns.barplot(x='CV_R_squared', y='Model', data=performance_df, palette='viridis')
plt.title('Cross-Validation R²', fontsize=14, fontweight='bold')
plt.xlabel('CV R²')

plt.tight_layout()
plt.show()

# %%
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import pandas as pd

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=45),
            'params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'colsample_bytree': [0.3, 0.5, 0.7],
                'alpha': [0, 5, 10]
            }
        },
        'MLP Regressor': {
            'model': MLPRegressor(max_iter=1000, random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['rbf', 'linear', 'poly'],
                'C': [0.1, 1.0, 10],
                'epsilon': [0.01, 0.1, 0.2]
            }
        },
        'KNN Regressor': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        },
        'AdaBoost': {
            'model': AdaBoostRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        },
        'Extra Trees': {
            'model': ExtraTreesRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Bagging Regressor': {
            'model': BaggingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_samples': [0.5, 0.7, 1.0],
                'max_features': [0.5, 0.7, 1.0]
            }
        }
    }
    
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    
    for algo_name, config in algos.items():
        try:
            print(f"Training {algo_name}...")
            gs = GridSearchCV(config['model'], config['params'], cv=cv, 
                            return_train_score=False, n_jobs=-1, verbose=0)
            gs.fit(X, y)
            scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_
            })
            print(f"✓ {algo_name} completed - Best Score: {gs.best_score_:.4f}")
        except Exception as e:
            print(f"✗ {algo_name} failed")
    
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Run GridSearchCV
print("Starting GridSearchCV for all models...\n")
results_df = find_best_model_using_gridsearchcv(X_train, Y_train)

# Display results sorted by best score
results_df = results_df.sort_values('best_score', ascending=False)
print("\n" + "="*80)
print("GRIDSEARCHCV RESULTS")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Show best model
best_model = results_df.iloc[0]
print(f"\n🏆 Best Model: {best_model['model']}")
print(f"   Best CV Score: {best_model['best_score']:.4f}")
print(f"   Best Parameters: {best_model['best_params']}")

# %%
