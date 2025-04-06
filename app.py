from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import base64
from kneed import DataGenerator, KneeLocator


# Use the Agg backend for Matplotlib to handle plotting without a GUI
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Configure the upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check for uploaded file
        uploaded_file = request.files.get('file_upload')
        
        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_location)
            flash(f"File uploaded successfully to {file_location}", 'success')
            
            try:
                # Process the uploaded file
                analysis_results, plots, gender_pie_charts = process_file(file_location)
                return render_template('results.html', analysis_results=analysis_results, plots=plots, gender_pie_charts=gender_pie_charts)
            except Exception as e:
                flash(f"Error processing file: {e}", 'error')
                print(f"Error processing file: {e}")
        else:
            flash("Invalid file type. Please upload a CSV file.", 'error')

    return render_template('index.html')

def process_file(file_path):
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError("Error reading the CSV file: " + str(e))

    # Check if required columns are present
    required_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'CustomerID', 'Gender']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("CSV file is missing required columns: " + ", ".join(required_columns))

    # Select the desired features for clustering
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate WCSS for different k values (Elbow Method)
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    # Perform KMeans clustering to find the optimal number of clusters
    optimal_k = determine_optimal_clusters(wcss)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
    df['cluster'] = y_kmeans

    # Analyze the characteristics of each customer segment
    segment_summary = df.groupby('cluster').agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'CustomerID': 'count'
    }).rename(columns={
        'Age': 'Mean Age',
        'Annual Income (k$)': 'Mean Annual Income (k$)',
        'Spending Score (1-100)': 'Mean Spending Score (1-100)',
        'CustomerID': 'Number of Customers'
    })

    plots = []

    # Histogram plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    for i, col in enumerate(X.columns):
        sns.histplot(X[col], ax=axs[i])
        axs[i].set_title('Histogram of ' + col)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.getvalue()).decode('utf8'))
    plt.close()

    # Elbow Method Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
    plt.xlabel("K Value")
    plt.xticks(range(1, 11))
    plt.ylabel("WCSS")
    plt.title("Elbow Method For Optimal k")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.getvalue()).decode('utf8'))
    plt.close()

    # Scatter plot of clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue="cluster",
                    palette=sns.color_palette("tab10", n_colors=optimal_k), legend='full', data=df, s=60)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Spending Score vs Annual Income Clusters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.getvalue()).decode('utf8'))
    plt.close()

    # 3D Plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(optimal_k):
        ax.scatter(df.Age[df.cluster == i], df["Annual Income (k$)"][df.cluster == i], df["Spending Score (1-100)"][df.cluster == i], s=60, label=f'Cluster {i}')
    ax.view_init(35, 185)
    plt.xlabel("Age")
    plt.ylabel("Annual Income (k$)")
    ax.set_zlabel('Spending Score (1-100)')
    plt.title('3D Clustering Plot')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots.append(base64.b64encode(buf.getvalue()).decode('utf8'))
    plt.close()

    # Calculate gender ratio for each cluster and create pie charts
    gender_pie_charts = []
    for cluster in df['cluster'].unique():
        gender_counts = df[df['cluster'] == cluster]['Gender'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        ax.axis('equal')
        plt.title(f'Gender Ratio for Cluster {cluster}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        gender_pie_charts.append(base64.b64encode(buf.getvalue()).decode('utf8'))
        plt.close()

    return segment_summary, plots, gender_pie_charts

def determine_optimal_clusters(wcss):
    """ Determine the optimal number of clusters using the Elbow method. """
    # You can adjust the logic here if needed
    kneedle = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
    return kneedle.elbow

if __name__ == '__main__':
    app.run(debug=True)

