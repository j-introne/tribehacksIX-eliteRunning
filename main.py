import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SS
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier as KNN
import time

# Read in the data on running mechanics and efficiency
def read_data():
    data = pd.read_csv("./data/Data.csv")
    data_num = data.drop(columns=["Subject.Code", "Group", "Speed", "Shoe", "Footstrike"])
    data_num_notNA = data_num.dropna(inplace=False)
    data_notNA = data.dropna(inplace=False)
    return data_notNA, data_num_notNA

# Return the features and target values based on the desired target name
def get_xy(target):
    data_notNA, data_num_notNA = read_data()
    X = data_num_notNA[["Running.Economy", "Mass", "Age", "Height"]].values
    y = data_notNA[target].values
    return X, y

if __name__ == "__main__":
    # Create the model to predict 10k times
    X, y = get_xy("10K.PB")

    # We need to normalize the data for lasso regression
    regress_ss = SS()
    Xscaled = regress_ss.fit_transform(X)

    # Fit the scaled data to the lasso regressor using the optimal cost hyperparameter
    las_reg = Lasso(alpha=0.04474747474747475)
    las_reg.fit(Xscaled, y)

    # Create the model to predict if a person's running efficiency is elite
    X, y = get_xy("Group")

    # We need to normalize the data again since KNN is based off euclidean distances
    knn_ss = SS()
    Xscaled = knn_ss.fit_transform(X)

    # Fit the scaled data to the KNN model using the optimal hyperparameter
    knn = KNN(n_neighbors=5, weights="distance")
    knn.fit(Xscaled, y)

    # Ask for the basic metrics of the user
    running_econ = float(input("Input your running economy (mL O2 / kg / km): "))
    mass = float(input("Input your mass (kg): "))
    age = float(input("Input your age (years): "))
    height = float(input("Input your height (cm): "))

    # Make an initial prediction for the user's 10k time
    predictions = [las_reg.predict(regress_ss.transform([[running_econ, mass, age, height]]))[0]]

    print("\nPredicted 10k Time: ", predictions[0])
    print("Are you elite: ", 1 == knn.predict(knn_ss.transform([[running_econ, mass, age, height]]))[0])

    time.sleep(2)

    # Based on the predicted time, provide a general outline for a training program to improve the user
    print("\nConsider the following training plan:\n")
    if predictions[0] > 55:
        # Novice level
        print("Run 2-3 miles every other day. Have one rest day, cross train one day, and do strength training the other days.")
        for i in range(9):
            # Interesting expression to see the most improvement in fitness in the middle of the training block
            running_econ -= 5/np.abs(7/2-i)
            predictions.append(las_reg.predict(regress_ss.transform([[running_econ, mass, age, height]]))[0])
    elif predictions[0] > 40:
        # Intermediate level
        print("Run 3-5 miles for five days of the week where one or two days includes running near goal 10k pace.",
              "\nRest or cross train for the other two days, and strength train two or three times a week.")
        for i in range(9):
            running_econ -= 3/np.abs(7/2-i)
            predictions.append(las_reg.predict(regress_ss.transform([[running_econ, mass, age, height]]))[0])
    else:
        # Advanced level
        print("Run 5-8 miles fives days a week where two or three include threshold work or paces near goal 10k pace.",
              "\nOne day should be a dedicated long run of more than 8 miles. Cross train one day, and do strength training three days.")
        for i in range(9):
            running_econ -= 1/np.abs(7/2-i)
            predictions.append(las_reg.predict(regress_ss.transform([[running_econ, mass, age, height]]))[0])

    time.sleep(2)

    # Graph the predicted 10k times over 10 weeks
    fig, ax = plt.subplots()
    ax.plot(range(1,11), predictions)
    ax.scatter(range(1,11), predictions)
    ax.set_title("10k Prediction Over 10 Weeks")
    ax.set_ylabel("10k Prediction Time")
    ax.set_xlabel("Week Number")
    plt.show()

    # Provide the final potential time and whether the metrics are elite
    print("\nFinal 10k Prediction: ", predictions[-1])
    print("\nCould you become elite!?")

    if 1 == knn.predict(knn_ss.transform([[running_econ, mass, age, height]]))[0]:
        print("Yes!")
    else:
        print("Not in 10 weeks, but never stop dreaming!")