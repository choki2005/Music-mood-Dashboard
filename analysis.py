import pandas as pd

# Load the transformed dataset
df = pd.read_csv("music_mood_transformed.csv")

# Select the music importance scores
importance = df["Music_Importance_Score"]

print("Music Importance Analysis")
print("-------------------------")

print("Mean:", importance.mean())
print("Median:", importance.median())
print("Mode:", importance.mode()[0])
print("Minimum:", importance.min())
print("Maximum:", importance.max())
print("Standard Deviation:", importance.std())


# Listening frequency analysis
frequency = df["Listening_Frequency_Score"]

print("\nListening Frequency Analysis")
print("---------------------------")

print("Mean:", frequency.mean())
print("Median:", frequency.median())
print("Mode:", frequency.mode()[0])
print("Minimum:", frequency.min())
print("Maximum:", frequency.max())
print("Standard Deviation:", frequency.std())


# Daily listening hours analysis
hours = df["Listening_Hours_Score"]

print("\nDaily Listening Hours Analysis")
print("------------------------------")

print("Mean:", hours.mean())
print("Median:", hours.median())
print("Mode:", hours.mode()[0])
print("Minimum:", hours.min())
print("Maximum:", hours.max())
print("Standard Deviation:", hours.std())


# Q10: Music type changes depending on mood
print("\nMusic Type Change by Mood")
print("-------------------------")

print(df["10. Does the type of music you listen to change depending on your mood?"].value_counts())


# Q10 numerical analysis
mood_change = df["Mood_Change_Score"]

print("\nMusic Type Change Depending on Mood Analysis")
print("---------------------------------------------")

print("Mean:", mood_change.mean())
print("Median:", mood_change.median())
print("Mode:", mood_change.mode()[0])
print("Minimum:", mood_change.min())
print("Maximum:", mood_change.max())
print("Standard Deviation:", mood_change.std())



# Q1: Age Group Analysis
print("\nAge Group Analysis")
print("------------------")

q1 = df["1. What is your age group?"]

print(q1.value_counts())

print("\nPercentages:")
print(q1.value_counts(normalize=True).mul(100).round(2))


# Q2: Gender Analysis
print("\nGender Analysis")
print("---------------")

q2 = df["2. What is your gender?"]

print(q2.value_counts())

print("\nPercentages:")
print(q2.value_counts(normalize=True).mul(100).round(2))


# Q11: How Music Generally Affects Mood
print("\nHow Music Generally Affects Mood")
print("--------------------------------")

q11 = df["11. How does music generally affect your mood?"]

print(q11.value_counts())

print("\nPercentages:")
print(q11.value_counts(normalize=True).mul(100).round(2))


# Q12: Genre Usually Listened to When Relaxed
print("\nGenre Usually Listened to When Relaxed")
print("--------------------------------------")

q12 = df[
    "12. Which music genre do you usually listen to when you want to feel relaxed?"
]

print(q12.value_counts())

print("\nPercentages:")
print(q12.value_counts(normalize=True).mul(100).round(2))


# Q13: Genre Usually Listened to When Energetic
print("\nGenre Usually Listened to When Energetic")
print("-----------------------------------------")

q13 = df[
    "13. Which music genre do you usually listen to when you want to feel energetic?"
]

print(q13.value_counts())

print("\nPercentages:")
print(q13.value_counts(normalize=True).mul(100).round(2))

# Q4: Music Importance Percentage Analysis

print("\nQ4: Music Importance Percentage Analysis")
print("----------------------------------------")

q4 = df["4. How important is music in your daily life?"]

print("\nFrequencies:")
print(q4.value_counts())

print("\nPercentages:")
print(q4.value_counts(normalize=True).mul(100).round(2))

# Q14: Relationship Analysis
print("\nQ14: Relationship Between Listening Frequency and Daily Listening Hours")
print("---------------------------------------------------------------------")

correlation = df["Listening_Frequency_Score"].corr(
    df["Listening_Hours_Score"]
)

print("Pearson Correlation:", correlation)