

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
import optuna

df = pd.read_csv("/content/sample_data/train.csv")

df1 = df.drop(["Podcast_Name", "Episode_Title", "id"], axis=1)

df1['Total_Popularity'] = df1['Host_Popularity_percentage'] + df1['Guest_Popularity_percentage']
df1['Ads_per_Minute'] = df1['Number_of_Ads'] / (df1['Episode_Length_minutes'] + 1)

for col in ['Episode_Length_minutes', 'Total_Popularity', 
            'Guest_Popularity_percentage', 'Host_Popularity_percentage']:
    df1[col].fillna(
        pd.Series(
            df1[col].dropna().sample(
                n=df1[col].isna().sum(), replace=True
            ).values, index=df1[df1[col].isna()].index
        ),
        inplace=True
    )

percentile_99 = np.percentile(df1["Episode_Length_minutes"], 99)
df1 = df1[df1["Episode_Length_minutes"] <= percentile_99]

df1_encoded = pd.get_dummies(
    df1, columns=['Genre', 'Publication_Time', 'Episode_Sentiment', 'Publication_Day'],
    drop_first=True
)

df1_encoded.fillna({"Number_of_Ads": 0, "Ads_per_Minute": 0}, inplace=True)

df1_encoded.drop_duplicates(inplace=True)
counts = df1_encoded['Number_of_Ads'].value_counts()
values_to_keep = counts[counts >= 5].index
df1_encoded = df1_encoded[df1_encoded['Number_of_Ads'].isin(values_to_keep)]


y = df1_encoded["Listening_Time_minutes"]
X = df1_encoded.drop(columns=["Listening_Time_minutes"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
def evaluate_model(name, model):
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")

def objective(trial):
    rf = RandomForestRegressor(
        n_estimators=trial.suggest_int("rf_n_estimators", 100, 300),
        max_depth=trial.suggest_int("rf_max_depth", 3, 20),
        random_state=42
    )
    xgb = XGBRegressor(
        n_estimators=trial.suggest_int("xgb_n_estimators", 100, 300),
        max_depth=trial.suggest_int("xgb_max_depth", 3, 15),
        learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
        random_state=42
    )
    final_type = trial.suggest_categorical("final_estimator", ["ridge", "lasso", "svr"])
    if final_type == "ridge":
        final_est = Ridge(alpha=trial.suggest_float("ridge_alpha", 0.1, 10.0))
    elif final_type == "lasso":
        final_est = Lasso(alpha=trial.suggest_float("lasso_alpha", 0.01, 1.0))
    else:
        final_est = SVR(C=trial.suggest_float("svr_C", 0.1, 10.0), kernel="rbf")

    stack = StackingRegressor(estimators=[('rf', rf), ('xgb', xgb)], final_estimator=final_est)
    score = cross_val_score(stack, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    return -score.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
best = study.best_params

rf = RandomForestRegressor(
    n_estimators=best['rf_n_estimators'], max_depth=best['rf_max_depth'], random_state=42
)
xgb = XGBRegressor(
    n_estimators=best['xgb_n_estimators'], max_depth=best['xgb_max_depth'],
    learning_rate=best['xgb_learning_rate'], random_state=42
)
ridge = Ridge(alpha=best['ridge_alpha'])

stack = StackingRegressor(estimators=[('rf', rf), ('xgb', xgb)], final_estimator=ridge)
stack.fit(X_train, y_train)

evaluate_model("Stacked RF + XGB (Tuned)", stack)

df_test = pd.read_csv("/content/sample_data/test.csv")
test_ids = df_test['id'].copy()

df_test = df_test.drop(["id", "Podcast_Name", "Episode_Title"], axis=1)
df_test["Total_Popularity"] = df_test["Host_Popularity_percentage"] + df_test["Guest_Popularity_percentage"]
df_test["Ads_per_Minute"] = df_test["Number_of_Ads"] / (df_test["Episode_Length_minutes"] + 1)

df_test.fillna({"Number_of_Ads": 0, "Ads_per_Minute": 0, "Total_Popularity": 0}, inplace=True)

df_test_encoded = pd.get_dummies(
    df_test, columns=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], drop_first=True
)
df_test_encoded = df_test_encoded.reindex(columns=X.columns, fill_value=0)

for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Host_Popularity_percentage']:
    df_test_encoded[col].fillna(
        pd.Series(
            df_test_encoded[col].dropna().sample(
                n=df_test_encoded[col].isna().sum(), replace=True
            ).values, index=df_test_encoded[df_test_encoded[col].isna()].index
        ),
        inplace=True
    )

stacked_predictions = stack.predict(df_test_encoded)
submission_df = pd.DataFrame({'id': test_ids, 'Listening_Time_minutes': stacked_predictions})
submission_df.to_csv('Submission.csv', index=False)
print("Submission.csv saved successfully!")
