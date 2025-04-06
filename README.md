# 📊 Segmentation and Insights Generation

This project is a web-based data analysis and visualization tool built using **Flask**, which allows users to upload a customer dataset (CSV format), perform **KMeans clustering**, and get **automated insights** in the form of tables and visual plots.

## 🚀 Features

✅ Upload CSV file through a simple web UI  
✅ Automated Elbow Method to find optimal clusters  
✅ Perform Customer Segmentation using KMeans  
✅ Summary table showing mean values and cluster sizes  
✅ Visualizations:
- Histogram distribution of input features
- Elbow method plot
- Scatter plot of clusters
- 3D clustering plot
- Gender ratio pie charts per cluster  
✅ Flash messages for success/error feedback  
✅ Easy to use, extend, and deploy  


## 📝 Input Format (CSV)

The uploaded CSV **must contain** the following columns:

| Column Name                | Description                        |
|---------------------------|------------------------------------|
| `CustomerID`              | Unique customer ID                 |
| `Gender`                  | Customer gender (Male/Female)      |
| `Age`                     | Age of customer                    |
| `Annual Income (k$)`      | Annual income in thousands         |
| `Spending Score (1-100)`  | Spending score on a scale of 1-100 |

📌 **Note:** Make sure column names match exactly as above (including case and spaces).


## 📦 Installation and Setup

### 🔧 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/Segmentation-and-Insights-Generation.git

cd Segmentation-and-Insights-Generation

### 📥 2. Install dependencies

pip install -r requirements.txt

### ▶️ 3. Run the Flask app

python app.py

### 🌐 4. Visit in browser

http://127.0.0.1:5000/


### 🧠 How It Works

1.Upload CSV via the web interface.

2.App performs preprocessing, including:
  
  -Validating required columns
  
  -Scaling features

3.Elbow Method determines optimal clusters (k).

4.KMeans Clustering segments the data.

5.Outputs include:
  
  -Summary table with mean age, income, spending score per cluster
  
  -Multiple visualizations including 2D, 3D, and pie charts


### 🛠️ Tech Stack
1.Python 3

2.Flask - Web framework

3.Pandas - Data manipulation

4.Matplotlib / Seaborn - Plotting

5.Scikit-learn - KMeans clustering

6.kneed - To find the Elbow point

7.HTML + Bootstrap - Frontend styling


### ✨ Highlights

💡 Automatically chooses the number of clusters using KneeLocator

📉 Uses visual storytelling to support data-driven decisions

🔄 Scalable for different kinds of customer datasets (retail, ecommerce, etc.)

📤 User-friendly CSV upload system


### 🧪 Use Cases

1.Retail Customer Segmentation

2.Marketing Target Group Identification

3.Business Intelligence Dashboards

4.Ecommerce Customer Clustering
