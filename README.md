# Predictions of the overall satisfaction of an airline passenger based on different flight experiences using decision trees and testing the models' robustness to label noise

## Project Title and Purpose
This project applies tree based machine learning methods to predict airline passenger satisfaction (Satisfied vs. Neutral or Dissatisfied) based on various flight experience factors. Additionally, it includes a controlled experiment to test the robustness of these models against training data label noise, simulating scenarios like corrupted datasets or incorrect survey replies.

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit learn (Decision Trees, Random Forests, GridSearchCV)
* **Visualization:** Matplotlib, Seaborn

## Technical Decisions
* **Data Preprocessing:** Removed irrelevant unique identifiers and flight delay variables to reduce noise. Categorical predictors were transformed using one hot encoding.
* **Model Selection:** Three specific models were trained for comparison: an Unpruned Classification Tree, a Pruned Classification Tree, and a Random Forest.
* **Hyperparameter Tuning:** Cost complexity pruning was tuned for the Pruned Tree using 10 fold Cross Validation. Random Forest hyperparameters were also optimized using GridSearchCV to balance training time and performance.
* **Robustness Experiment:** Created copies of the training dataset with randomly flipped output labels at increasing noise levels (10 percent, 20 percent, 30 percent) while keeping the test set unchanged to evaluate model stability under data corruption.

## Key Findings and Challenges
* The Unpruned Tree severely overfit the training data (100 percent training accuracy, depth of 42), achieving a baseline test accuracy of 94.59 percent.
* Cost complexity pruning successfully reduced model complexity (depth of 26) and slightly improved the test accuracy to 95.92 percent.
* The Random Forest model achieved the highest baseline test accuracy at 96.48 percent on the clean dataset.
* **Robustness Insight:** The experiment revealed a critical trade off between complexity and stability; while the complex Random Forest performed best on clean data, the regularised Pruned Classification Tree proved to be the most stable and robust model when significant label noise was introduced.

## Installation Instructions
1. Clone the repository: `git clone <repository_url>`
2. Install the required dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run the Jupyter Notebook to view the data processing, model evaluation, and robustness experiment.
