import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load Data
df = pd.read_csv('final data.csv')

#Split Data
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
clf1 = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('lr', LogisticRegression(max_iter=1000))
])

# Random Forest
clf2 = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Support Vector Machine (SVM)
clf3 = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('svm', SVC(probability=True, kernel='linear', random_state=42))
])

# Create & Train Voting Classifier
print("Training Ensemble Model...")
eclf = VotingClassifier(estimators=[
    ('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='soft')

eclf.fit(X_train, y_train)

#joblib.dump(eclf, 'forensic_model.pkl')
#print("Model saved as forensic_model.pkl")

y_pred = eclf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))