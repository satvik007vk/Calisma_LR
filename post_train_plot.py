import matplotlib.pyplot as plt
import seaborn as sns
from train_and_evaluate import load_train_test_data, train_mlr, get_mlr_coefficients, evaluate_model, \
    shap_explainability, train_xgboost, train_rf

# Load data again (to get df_train)
#TODO - improve the function definition to only choose predictand among the given values
predictand: str = 'both'  # 'lwp' or 'clf' or 'both'

df_train, df_test, X_train, y_train, X_test, y_test, predictors, predictands = load_train_test_data(
    predictand=predictand)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# TODO - solve the inconsistent no of samples error
mlr_model = train_mlr(X_train, y_train)
mlr_coefficients = get_mlr_coefficients(mlr_model, predictors, predictands)
evaluate_model(mlr_model, X_train, y_train, 'Multi linear regression')

xgb_model = train_xgboost(X_train, y_train)
evaluate_model(xgb_model, X_train, y_train, 'XGBoost')

# rf_model = train_rf(X_train, y_train)
# evaluate_model(rf_model, X_train, y_train, 'Random Forest')

#shap_explainability(mlr_model, X_train, y_train)

# ---- 1. Bar Plot of Coefficients ----
def plot_coefficients(mlr_coefficients):
    plt.figure(figsize=(10, 5))
    mlr_coefficients.T.plot(kind='bar', figsize=(12, 6))
    plt.title("Multi-Linear Regression Coefficients")
    plt.xlabel("Predictors")
    plt.ylabel("Coefficient Value")
    plt.legend(title="Predictands")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.show()


# ---- 2. Scatter Plots with Regression Lines ----
def plot_scatter_relationships(df_train, mlr_coefficients):
    top_predictors = mlr_coefficients.abs().mean(axis=0).nlargest(3).index.tolist()

    for predictor in top_predictors:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot predictor vs clf
        sns.regplot(x=df_train[predictor], y=df_train['clf'], ax=axes[0], scatter_kws={'alpha': 0.5})
        axes[0].set_title(f"Relationship between {predictor} and clf")
        axes[0].set_xlabel(predictor)
        axes[0].set_ylabel("clf")

        # Plot predictor vs lwp
        sns.regplot(x=df_train[predictor], y=df_train['lwp'], ax=axes[1], scatter_kws={'alpha': 0.5})
        axes[1].set_title(f"Relationship between {predictor} and lwp")
        axes[1].set_xlabel(predictor)
        axes[1].set_ylabel("lwp")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_coefficients(mlr_coefficients)
#    plot_scatter_relationships(df_train, mlr_coefficients)
