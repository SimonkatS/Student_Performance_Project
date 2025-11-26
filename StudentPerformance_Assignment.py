import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# GITHUB LINK TO CLONE: https://github.com/SimonkatS/Student_Performance_Project
# Dataset : https://www.kaggle.com/code/nabeelqureshitiii/student-performance-prediction/input

# TO READ PROPERLY HAVE: -FOLDER BEFORE r"...\" OPEN IN VSCODE -HAVE THE SAME FOLDER/FILE NAMES IF YOU RUN ON DIFFERENT DEVICE
try:
    dataset = pd.read_csv("student_performance.csv")
except FileNotFoundError:
    # Fallback
    try:
        dataset = pd.read_csv(r"StudentPerformance_Assignment\student_performance.csv")
    except FileNotFoundError:
        print("Error: File not found. Please make sure 'student_performance.csv' is in the same folder.")
        exit()
dataset = dataset.sample(n=50000, random_state=42)
# print(dataset.describe())  # helps to see some data from the csv
features = ["weekly_self_study_hours", "attendance_percentage", "class_participation", "total_score"]
X = dataset[features]  ## droping 'grade' because its not numerical
scaler = StandardScaler()  #normalizing
X_scaled = scaler.fit_transform(X)


##  ELBOW METHOD TO SEE HOW MANY CLUSTERS ARE OPTIMAL  (PREVIOUS RESULTS SUGGEST K=3or4 ) Very slow, does kmeans 10 times
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

while True:
    try:
        k = int(input("Enter the number of clusters (e.g., 2, 3...): "))
        if k > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")


kmean = KMeans(n_clusters=k, random_state=42)  #kmeans is the simplest, works well with the dataset, and we have learned it in class
kmean.fit(X_scaled)
dataset['Cluster'] = kmean.labels_
print("Model training complete.")


# STATISTICAL SUMMARY
print("\n--- Statistical Summary for Each Cluster ---")
print(dataset.groupby('Cluster')[features].describe().T.round(2))


# CLUSTER VISUALIZATION
print("\nGenerating Pairplot... (This shows relationships between all variables)")
# 'hue' colors the dots by Cluster
# 'vars' ensures we only plot the relevant columns
# 'diag_kind' makes the diagonal graphs smooth curves instead of bars
sns.pairplot(dataset, hue='Cluster', vars=features, palette='bright', diag_kind='kde')
plt.show()


# NEAREST NEIGHBORS
nn_model = NearestNeighbors(n_neighbors=4, algorithm='auto')
nn_model.fit(X_scaled)

# NEW STUDENT ENTRY LOOP 

print("\n--- New student entry ---")
 
while True:
    user_choice = input("\nDo you want to enter a new student? (y/n): ").lower()
    if user_choice != 'y':
        break

    try:
        print("Enter student details:")
        f1 = float(input(f" - {features[0]}: "))
        f2 = float(input(f" - {features[1]}: "))
        f3 = float(input(f" - {features[2]}: "))
        f4 = float(input(f" - {features[3]}: "))

        # Prepare the data
        new_data = np.array([[f1, f2, f3, f4]])
        # Scale it using the existing scaler
        new_data_scaled = scaler.transform(new_data)

        # Predict Cluster
        predicted_cluster = kmean.predict(new_data_scaled)[0]
        print(f"\n---> This student belongs to Group (Cluster): {predicted_cluster}")

        # Find Neighbors
        distances, indices = nn_model.kneighbors(new_data_scaled)
        
        print(f"---> The 3 most similar students in the database are:")
        similar_students = dataset.iloc[indices[0]]
        print(similar_students[features + ['Cluster']])

    except ValueError:
        print("Invalid input! Please enter numbers only.")
    except Exception as e:
        print(f"An error occurred: {e}")

print("Exiting program.")

