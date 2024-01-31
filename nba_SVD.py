import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""First order is to download the excel file onto the local drive of whoever runs this script.
Then it will transfer the data from the excel file onto this python script."""

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


file_id = '1qfAykoXGr7CaLAEbj3Rqx4rN81jc2JJO'
destination = './predicting_nba_wins.xlsx'
download_file_from_google_drive(file_id, destination)
"""Task 1 is complete. Now whoever runs this should have the spreadsheet downloaded onto their drive."""
# Load spreadsheet
xl = pd.ExcelFile('./predicting_nba_wins.xlsx')

# Load a sheet into a DataFrame by its name
df1 = xl.parse('22-23')

# Set the display options.
pd.set_option('display.max_rows', None)      # None means no limit
pd.set_option('display.max_columns', None)   # None means no limit
pd.set_option('display.width', 200)          # Adjust as needed
pd.set_option('display.max_colwidth', None)  # None means no limit

# df is DataFrame
A = df1.loc[:, 'TS%':'Defensive Rating'].values
b = df1['Actual Wins'].values

# solve for x
u1, s1, v1 = np.linalg.svd(A)
# Convert s1 to a diagonal matrix
s1_diag = np.zeros((u1.shape[1], v1.shape[0]))
for i, s in enumerate(s1):
    s1_diag[i, i] = s


# Compute A from U, S, and V
A_actually = u1 @ s1_diag @ v1

# Compute the pseudoinverse of S
s1_diag_pinv = np.zeros_like(s1_diag.T)
np.fill_diagonal(s1_diag_pinv, 1 / s1[s1 != 0])

# Compute the pseudoinverse of A
A_pinv = v1.T @ s1_diag_pinv @ u1.T
x = A_pinv @ b

# Print the solution
print('\nSolution:', x)
print('')

# Define metric labels
metric_labels = ['TS%', 'Age', 'SRS', 'DeFG%', 'Defensive Rating']

# Calculate the absolute values of x
abs_x = np.abs(x)

# Calculate the sum of absolute values of x
abs_x_sum = np.sum(abs_x)

# Calculate the percentage weight for each metric
weights = (abs_x / abs_x_sum) * 100

# Print the weights for each metric
print("Weight percentage for each metric:")
for label, weight in zip(metric_labels, weights):
    print(f"{label}: {weight:.2f}%")

print('')

avg_differences = []

"""Predicting Wins for the 21-22 NBA Season"""

df2 = xl.parse('21-22')
A_22 = df2.loc[:, 'TS%': 'Defensive Rating'].values

predicted_wins = [np.dot(row, x) for row in A_22]
df2['Predicted Wins'] = predicted_wins

# Compute the difference between actual and predicted wins
df2['Difference'] = df2['Actual Wins'] - df2['Predicted Wins']

# Compute the average difference
average_difference = df2['Difference'].mean()
avg_differences.append(average_difference)

print(df2)
print('Average difference:', average_difference)
print('')


"""Predicting Wins for the 09-10 NBA Season"""

df3 = xl.parse('09-10')
A_10 = df3.loc[:, 'TS%': 'Defensive Rating'].values

predicted_wins = [np.dot(row, x) for row in A_10]
df3['Predicted Wins'] = predicted_wins

# Compute the difference between actual and predicted wins
df3['Difference'] = df3['Actual Wins'] - df3['Predicted Wins']

# Compute the average difference
average_difference = df3['Difference'].mean()
avg_differences.append(average_difference)

print(df3)
print('Average difference:', average_difference)
print('')


"""Predicting Wins for the 99-00 NBA Season"""

df4 = xl.parse('99-00')
A_00 = df4.loc[:, 'TS%': 'Defensive Rating'].values

predicted_wins = [np.dot(row, x) for row in A_00]
df4['Predicted Wins'] = predicted_wins

# Compute the difference between actual and predicted wins
df4['Difference'] = df4['Actual Wins'] - df4['Predicted Wins']

# Compute the average difference
average_difference = df4['Difference'].mean()
avg_differences.append(average_difference)

print(df4)
print('Average difference:', average_difference)
print('')


"""Predicting Wins for the 90-91 NBA Season"""

df5 = xl.parse('90-91')
A_91 = df5.loc[:, 'TS%': 'Defensive Rating'].values

predicted_wins = [np.dot(row, x) for row in A_91]
df5['Predicted Wins'] = predicted_wins

# Compute the difference between actual and predicted wins
df5['Difference'] = df5['Actual Wins'] - df5['Predicted Wins']

# Compute the average difference
average_difference = df5['Difference'].mean()
avg_differences.append(average_difference)

print(df5)
print('Average difference:', average_difference)
print('')


"""Predicting Wins for the 80-81 NBA Season"""

df6 = xl.parse('80-81')
A_81 = df6.loc[:, 'TS%': 'Defensive Rating'].values

predicted_wins = [np.dot(row, x) for row in A_81]
df6['Predicted Wins'] = predicted_wins

# Compute the difference between actual and predicted wins
df6['Difference'] = df6['Actual Wins'] - df6['Predicted Wins']

# Compute the average difference
average_difference = df6['Difference'].mean()
avg_differences.append(average_difference)

print(df6)
print('Average difference:', average_difference)
print('')


"""Predicting Wins for the 70-71 NBA Season"""

df7 = xl.parse('70-71')
A_71 = df7.loc[:, 'TS%': 'Defensive Rating'].values

predicted_wins = [np.dot(row, x) for row in A_71]
df7['Predicted Wins'] = predicted_wins

# Compute the difference between actual and predicted wins
df7['Difference'] = df7['Actual Wins'] - df7['Predicted Wins']

# Compute the average difference
average_difference = df7['Difference'].mean()
avg_differences.append(average_difference)

print(df7)
print('Average difference:', average_difference)
print('')


"""Predicting Wins for the 60-61 NBA Season"""

df8 = xl.parse('60-61')
A_61 = df8.loc[:, 'TS%': 'Defensive Rating'].values

predicted_wins = [np.dot(row, x) for row in A_61]
df8['Predicted Wins'] = predicted_wins

# Compute the difference between actual and predicted wins
df8['Difference'] = df8['Actual Wins'] - df8['Predicted Wins']

# Compute the average difference
average_difference = df8['Difference'].mean()
avg_differences.append(average_difference)

print(df8)
print('Average difference:', average_difference)
print('')

"""Graphing the average difference between predicted wins and actual wins through the years. 
This shows our model gradually breaking down the further back in time we go, 
showing how stats that predict success has changed over time."""

years = [2022, 2010, 2000, 1991, 1981, 1971, 1961]
fig, ax = plt.subplots()

# Plot the data
ax.plot(years, avg_differences, marker='o')

# Set the labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Average Differences')
ax.set_title('Difference between predicted wins and actual wins')

# Reverse the x-axis
ax.invert_xaxis()

# Display the plot
plt.show()