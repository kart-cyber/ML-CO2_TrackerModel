
#all the imports
import pandas as pd
import numpy as np
import joblib


from sklearn.model_selection import train_test_split #this command allows for the split of the training phase and the testing phase
from sklearn.pipeline import Pipeline #allows for us to utilize the pipeline itself
from sklearn.compose import ColumnTransformer # enables column-wise transformations (e.g., scaling numeric, encoding categorical) in one pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer #handles any missing values within the data
from sklearn.ensemble import RandomForestRegressor #allows for us to import a model called RandomForestRegressor which utilizes decsion trees to determine a final average output.
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error #utilizing metrics to determine the accuracy/error percentage of the prediction




#importing other sources

#









#Loading in our data
data = pd.read_csv("vehicle_emissions.csv") #reads the file and converts it to a pandas dataframe


data.columns = data.columns.str.strip()

print([repr(c) for c in data.columns])  # debug once

#creating the features and the target variable

#treat these as flashcards(question on front, answer behind)
#treat X as the front of the flashcard
X = data.drop(["CO2_Emissions"], axis = 1)
#treat Y as the back of the card(the answer)
y = data["CO2_Emissions"]


#splitting the list into numerical and catergorial features

#this utilizes columns that are already considered as numerical data(floats, ints, etc)
numerical_cols = ["Model_Year", "Engine_Size", "Cylinders", "Fuel_Consumption_in_City(L/100 km)", "Fuel_Consumption_in_City_Hwy(L/100 km)", "Fuel_Consumption_comb(L/100km)", "Smog_Level"]

catergorial_cols = ["Make", "Model", "Vehicle_Class", "Transmission"]

#start the pipeline with encoding
#if the category is missing a value, just input the average value
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('scaler',StandardScaler())
    
])

#if the category is missing a value, just input the most frequent as it requires a string
#the encoder allows for the string to be converted to binar
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


#Join the pipelines together

#requires the pipeline and the data it needs to combine. This is considered as the cleaning stage
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, catergorial_cols)
])

#this now combines everything into one pipeline as now we have all the data needed as well as the model that is going to be used to make predictions from this data
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor()),
])

#Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


#when "training"(learning) the model is given the front and back of the flashcard(like studying/memorizing terms before quizzing)\
#however when "testing"(quizzing), the model is only given the front of the flashcard(X_test) as it needs to predict and not rely on the actual value


#Training the model
pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)

#View the encoding that was done
encoded_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(catergorial_cols)
#print(encoded_cols)

#Evaluate model accuracy using average of squared differences(the lower the better)
mse = mean_squared_error(y_test,prediction)
rmse = np.sqrt(mse)

#returns decimal range from 0-1, the higher the value the better(serves a trendline)
r2 = r2_score(y_test,prediction)

#returns average of ABSOLUTE differences. The lower the better
mae = mean_absolute_error(y_test,prediction)

print(f"Model Performance: ")
print(f"R2 score: {r2}")
print(f"Root mean squared error: {rmse}")
print(f"Mean Absolute Error: {mae}")


print("Saving model...")
joblib.dump(pipeline, "C:/Users/Rampe/OneDrive/Desktop/mlpipelinetest/vehicle_emission_pipeline.joblib")
print("Saved successfully.")

