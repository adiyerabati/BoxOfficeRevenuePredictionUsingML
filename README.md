# Box Office Revenue Prediction Using ML

When a movie is produced, the director often aims to maximize its revenue. But is it possible to predict a movie's revenue based on its genre or budget? This guide demonstrates how to build a machine learning model to predict box office revenue using features such as genre, budget, and other related attributes.

## Importing Libraries and Dataset

Python provides powerful libraries that simplify data handling and allow us to execute complex tasks efficiently. Here's an overview of the libraries we'll use:

- **Pandas**: For loading and manipulating data in a tabular (2D array) format, with many built-in functions for analysis.
- **NumPy**: Enables fast computations and handling of large numerical arrays.
- **Matplotlib/Seaborn**: Used for creating data visualizations.
- **Scikit-learn (Sklearn)**: Provides a comprehensive toolkit for data preprocessing, model development, and evaluation.
- **XGBoost**: A high-performance gradient-boosting library that excels in predictive modeling tasks.

### Python Imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')
