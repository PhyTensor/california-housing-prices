import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Machine Learning Lifecycle""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Collection""")
    return


@app.cell
def _():
    import pandas as pd
    from pandas import DataFrame, Series


    def load_dataset(data_path: str) -> DataFrame:
        # Load the dataset
        df: DataFrame = pd.read_csv(data_path)

        return df
    return DataFrame, Series, load_dataset, pd


@app.cell
def _(DataFrame, load_dataset):
    df: DataFrame = load_dataset("housing.csv")
    # Display the first few rows of the DataFrame
    df.head()
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    # Check missin values
    df.isnull().sum()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Data Cleaning and Preprocessing

        LinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively.

        Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values')
        """
    )
    return


@app.cell
def _(DataFrame, pd):
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    def normalisation(data: DataFrame) -> None:
        # Normalisation/Feature standardisation - normalise numerical features using standard scalar
        scaler = StandardScaler()
        # select columns with numerical features
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns

        data[num_cols] = scaler.fit_transform(data[num_cols])


    def fill_missing_numerical(data: DataFrame) -> None:
        # Fill missing values for numerical features using median
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            # data[col].fillna(data[col].median(), inplace=True)
            data.fillna({col: data[col].median()}, inplace=True)


    def fill_missing_categorical(data: DataFrame) -> None:
        # Fill missing values for categorical features with the mode
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            # data[col].fillna(data[col].mode()[0], inplace=True)
            data.fillna({col: data[col].mode()}, inplace=True)


    def handle_nan_values(data: DataFrame) -> None:
        # Handle values with NaNs - only works with numeric data
        imputer = SimpleImputer(strategy='median')
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[num_cols] = imputer.fit_transform(data[num_cols])


    def convert_categorical(data: DataFrame) -> DataFrame:
        return pd.get_dummies(data, columns=['ocean_proximity'])
    return (
        SimpleImputer,
        StandardScaler,
        convert_categorical,
        fill_missing_categorical,
        fill_missing_numerical,
        handle_nan_values,
        normalisation,
    )


@app.cell
def _(
    DataFrame,
    convert_categorical,
    df,
    fill_missing_categorical,
    handle_nan_values,
):
    fill_missing_categorical(df)
    fill_missing_categorical(df)
    handle_nan_values(df)
    cdf: DataFrame = convert_categorical(df)
    # normalisation(df)
    # normalisation(cdf)
    return (cdf,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(cdf):
    print(cdf.head())
    cdf
    return


@app.cell
def _(mo):
    mo.md(r"""## Exploratory Data Analysis""")
    return


@app.cell
def _(cdf):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Distribution of target
    plt.figure(figsize=(10, 6))
    sns.histplot(cdf['median_house_value'], kde=True)
    plt.show()
    return plt, sns


@app.cell
def _(cdf, plt, sns):
    # Correlation matrix
    corr = cdf.corr()
    plt.figure(figsize=(11, 4))
    sns.heatmap(data=corr[abs(corr['median_house_value']) > 0.4], annot=True)
    plt.show()
    return (corr,)


@app.cell
def _(mo):
    mo.md(r"""## Feature Engineering and Selection""")
    return


@app.cell
def _(DataFrame, Series, cdf):
    # Create new features
    cdf['rooms_per_household'] = cdf['total_rooms']/cdf['households']
    cdf['bedrooms_per_room'] = cdf['total_bedrooms']/cdf['total_rooms']

    # Select features
    features: list = [
        'median_income', 'housing_median_age', 'rooms_per_household', 'bedrooms_per_room', 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND'
    ]

    X: DataFrame = cdf[features]
    y: Series = cdf['median_house_value']
    return X, features, y


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Model Selection

        """
    )
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds)
        print(f"{name}: RMSE = {rmse:.2f}, R^2 = {r2_score(y_test, preds):.2f}")
    return (
        LinearRegression,
        RandomForestRegressor,
        X_test,
        X_train,
        mean_squared_error,
        model,
        models,
        name,
        preds,
        r2_score,
        rmse,
        train_test_split,
        y_test,
        y_train,
    )


@app.cell
def _(mo):
    mo.md(r"""## Model Training and Hyperparameter Tuning""")
    return


@app.cell
def _(RandomForestRegressor, X_train, y_train):
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    return GridSearchCV, best_model, grid_search, param_grid


@app.cell
def _(mo):
    mo.md(r"""## Model Evaluation and Tuning""")
    return


@app.cell
def _(
    X_test,
    best_model,
    features,
    mean_squared_error,
    pd,
    plt,
    r2_score,
    y_test,
):
    from sklearn.metrics import mean_absolute_error

    final_preds = best_model.predict(X_test)

    print(f"Final RMSE: {mean_squared_error(y_test, final_preds):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, final_preds):.2f}")
    print(f"R^2: {r2_score(y_test, final_preds):.2f}")

    # Feature importance
    pd.Series(best_model.feature_importances_, index=features).plot(kind='barh')
    plt.savefig('reports/feature_importance.png')
    plt.show()
    return final_preds, mean_absolute_error


@app.cell
def _(best_model):
    import joblib

    # Save the model
    filename = 'model.sav'
    joblib.dump(best_model, filename)
    return filename, joblib


@app.cell
def _(mo):
    mo.md(r"""## Model Deployment with FastAPI and Docker""")
    return


@app.cell
def _(mo):
    mo.md(r"""## CI/CD Pipeline""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Model Monitoring and Maintenance""")
    return


if __name__ == "__main__":
    app.run()
