from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        train = pd.read_csv('train1.csv', encoding='utf-8')
        test = pd.read_csv('test1.csv', encoding='utf-8')
        
        X_train = train.drop("target", axis=1)
        y_train = train["target"]
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        classifier = RandomForestClassifier(n_estimators=30, random_state=42)
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_val)
        accuracy = classifier.score(X_val, y_val)
        
        report = classification_report(y_val, y_pred)
        
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.countplot(x="target", data=train)
        plt.title("Distribution of Target Variable")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.savefig("static/distribution_plot.png")
        
        return render_template('results.html', accuracy=accuracy, report=report)

if __name__ == '__main__':
    app.run(debug=True)
