from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import silhouette_score, make_scorer
import base64

app = Flask(__name__,template_folder = "templates")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('index'))
        if file:
            train = pd.read_csv(file, encoding='utf-8')
            X_train = train.drop('target', axis=1)

            param_grid = {
                'n_clusters': [160, 165, 170, 175],  
                'init': ['k-means++', 'random'],
                'max_iter': [300, 500, 1000] 
            }

            silhouette_scorer = make_scorer(silhouette_score)
            kmeans = GridSearchCV(KMeans(), param_grid, cv=5, scoring=silhouette_scorer)
            kmeans.fit(X_train)

            best_params = kmeans.best_params_
            best_kmeans = KMeans(**best_params)
            best_kmeans.fit(X_train)

            test = pd.read_csv('test.csv', encoding='utf-8')
            test_clusters = best_kmeans.predict(test)
            silhouette = silhouette_score(test, test_clusters)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(test['x'], test['y'], c=test_clusters, cmap='viridis')
            plt.title('Clustering Diagram')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.colorbar(label='Cluster')
            
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            
            image_str = "data:image/png;base64," + base64.b64encode(image_png).decode()
            
            return render_template('result.html', silhouette=silhouette)

if __name__ == '__main__':
    app.run(debug=True)