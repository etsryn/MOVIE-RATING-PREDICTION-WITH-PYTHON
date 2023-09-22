import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress certain warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the path to the movie dataset
file_path = 'IMDb_Movies_India.csv'
data = None

# Attempt to read the dataset using different encodings
encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252', 'unicode_escape']
for encoding in encodings_to_try:
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        break  # Exit the loop if successful
    except UnicodeDecodeError:
        continue  # Try the next encoding

# Check if the dataset was successfully loaded
if data is None:
    print("Failed to read the dataset with any encoding.")
else:
    print("Dataset loaded successfully...")

    # Handle missing values by filling them with defaults
    data.fillna({'Genre': 'Unknown', 'Director': 'Unknown', 'Actor 1': 'Unknown', 'Actor 2': 'Unknown', 'Actor 3': 'Unknown', 'Rating': 0}, inplace=True)
    print("Missing values filled with default values...")

    # Extract features and target variable
    X = data[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
    y = data['Rating']
    print("Features and target variable extracted...")

    # Preprocess categorical features using one-hot encoding with drop_first=True
    X = pd.get_dummies(X, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], drop_first=True)
    print("Categorical features preprocessed with one-hot encoding...")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets...")

    # Initialize and train the XGBoost regression model
    model = xgb.XGBRegressor(n_jobs=-1)
    model.fit(X_train, y_train)
    print("XGBoost model trained successfully...")

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    print("Predictions made on the test set...")

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print("Model evaluation completed...")

    # Print evaluation metrics
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')

# Create a scatter plot of predicted vs. actual ratings
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', label='Actual Ratings')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual Ratings vs. Predicted Ratings')
plt.legend()
plt.show()

# Create a histogram to visualize the distribution of ratings
plt.figure(figsize=(8, 6))
sns.histplot(y_test, bins=30, kde=True, color='blue', label='Actual Ratings')
sns.histplot(y_pred, bins=30, kde=True, color='red', label='Predicted Ratings')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.legend()
plt.title('Distribution of Actual and Predicted Ratings')
plt.show()

# Indicate that the prediction for the new movie is complete
print("Prediction for the new movie is finished.")
