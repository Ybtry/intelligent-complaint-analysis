Intelligent Complaint Analysis
This project aims to analyze consumer complaints to extract meaningful insights, identify common themes (topics), and potentially classify complaints for better understanding and response.

Project Structure
The project is organized into the following directories and files:

intelligent_complaint_analysis/
├── .venv/                      # Python virtual environment
├── data/                       # Stores raw and processed data
│   ├── complaints.csv          # Original large dataset (NOT version controlled)
│   └── processed/              # Contains processed data and models
├── notebooks/                  # Jupyter Notebooks for analysis and model development
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_text_representation.ipynb
├── src/                        # Python scripts for modular code 
├── .gitignore                  # Specifies files/folders to be ignored by Git
├── README.md                   # Project overview and instructions
└── requirements.txt            # Lists all Python package dependencies

Setup and Installation
To set up and run this project on your local machine, follow these steps:

Clone the Repository:
Open your Git Bash terminal (or PowerShell/Command Prompt) and navigate to the directory where you want to store your project (e.g., C:\Users\YourUsername\Documents). Then clone the repository:

cd /c/Users/YourUsername/Documents
git clone https://github.com/Ybtry/intelligent-complaint-analysis.git
cd intelligent_complaint_analysis

Create and Activate Virtual Environment:
It's highly recommended to use a virtual environment to manage project dependencies.

python -m venv .venv

Activate the virtual environment:

On Windows (Git Bash/Command Prompt/PowerShell):

source ./.venv/Scripts/activate

You should see (.venv) at the beginning of your terminal prompt.

Install Dependencies:
With the virtual environment activated, install all required Python packages:

pip install -r requirements.txt

Download and Place complaints.csv:
The original complaints.csv dataset is too large for GitHub. You must manually download it and place it in the data/ directory.

Download complaints.csv from its original source.

Place the downloaded file into intelligent_complaint_analysis/data/.

The full path should be intelligent_complaint_analysis/data/complaints.csv.

Usage and Progress
This project is structured into several tasks, each with its dedicated Jupyter Notebook.

Exploratory Data Analysis (EDA) & Preprocessing (notebooks/task1_eda_preprocessing.ipynb)

Purpose: Initial data understanding, cleaning, and preparation.

Key Operations:

Efficiently loaded the large complaints.csv by selecting only Product and Consumer complaint narrative columns.

Performed initial data inspection (.info(), .head(), .isnull().sum(), .describe()).

Analyzed product distribution and narrative lengths.

Filtered data to focus on relevant product categories.

Cleaned consumer complaint narratives (lowercase, removed boilerplate, special characters).

Output: data/filtered_complaints.csv (cleaned and filtered dataset).

Text Representation (notebooks/task2_text_representation.ipynb)

Purpose: Convert raw text narratives into numerical features suitable for machine learning.

Key Operations:

Loaded data/filtered_complaints.csv.

Applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to the cleaned_narrative column.

Generated a TF-IDF matrix with 5000 features.

Output:

data/processed/tfidf_matrix.pkl (the numerical feature matrix).

data/processed/tfidf_vectorizer.pkl (the trained TF-IDF model).

data/processed/df_filtered_with_indices.pkl (the filtered DataFrame with original indices).

Task 3: Topic Modeling (notebooks/task3_topic_modeling.ipynb)
Status: In Progress / Next Step

Purpose: Discover latent themes or topics within the consumer complaints.

Key Operations:

Will load the TF-IDF matrix and vectorizer from Task 2.

Will train a Non-negative Matrix Factorization (NMF) model.

Will extract and display top words for each identified topic.

Will assign the dominant topic to each complaint.

Output: data/processed/df_with_topics.pkl (DataFrame with assigned topics).

Running the Notebooks
Open VS Code and open the intelligent_complaint_analysis folder (File > Open Folder...).

Open the desired .ipynb file from the notebooks/ directory.

Select your virtual environment as the kernel: In the top-right of the notebook pane, click on the Python interpreter name (e.g., "Python 3.x.x") and select the (.venv) environment.

Run all cells in the notebook.