import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.arrays import categorical
import seaborn as sb
from termcolor import colored as cl

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge 
from sklearn.linear_model import ElasticNet 

from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2

colors = ['rosybrown', 'lightcoral', 'burlywood', 'darkred', 'salmon', 'sienna', 'tomato', 'sandybrown', 'peachpuff', 'lightsalmon']
r2_scores = {}
explained_variance = {}

sb.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (10, 5)

def prepData():

    trainSet = pd.read_csv('HouseData.csv')
    trainSet.set_index("Id", inplace = True)
    trainSet.dropna(inplace = True, axis=1)
    trainData = trainSet.select_dtypes([np.number])

    return trainData

def scatterPlot(dataFrame, variableNum):
    scatter_df = dataFrame.drop('SalePrice', axis = 1)
    cols = dataFrame.columns

    plot = sb.scatterplot(cols[variableNum], 'SalePrice', data = dataFrame, color = colors[variableNum % 10], edgecolor = 'white', s = 100)
    plt.title('{} / Sale Price'.format(cols[variableNum]), fontsize = 16)
    plt.xlabel('{}'.format(cols[variableNum]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    name = 'scatter' + str(variableNum) + '.png'
    plt.savefig(name)
    plt.show()

def distribution(data):

    sb.distplot(data['SalePrice'], color = 'r')
    plt.title('Sale Price Distribution', fontsize = 16)
    plt.xlabel('Sale Price', fontsize = 14)
    plt.ylabel('Frequency', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    plt.savefig('distplot.png')
    plt.show()

def plot(data):

    # Plot correlations between each category in heatmap
    sb.heatmap(data.corr(), annot = True, cmap = 'magma')

    plt.savefig('heatmap.png')
    plt.show()

    # Make scatterplots of each category and sale price
    for variable in range(33):
        scatterPlot(data, variable)
    
    # Make sale price distribution plot
    distribution(data)

def ols(X_train, X_test, Y_train, Y_test):

    ols = LinearRegression()
    ols.fit(X_train, Y_train)
    yHat = ols.predict(X_test)

    r2_scores["OLS"] = r2(Y_test, yHat)
    explained_variance["OLS"] = evs(Y_test, yHat)


def ridge(X_train, X_test, Y_train, Y_test):
    
    ridge = Ridge(alpha = 0.5)
    ridge.fit(X_train, Y_train)
    yHat = ridge.predict(X_test)

    r2_scores["Ridge"] = r2(Y_test, yHat)
    explained_variance["Ridge"] = evs(Y_test, yHat)

def lasso(X_train, X_test, Y_train, Y_test):

    lasso = Lasso(alpha = 0.01)
    lasso.fit(X_train, Y_train)
    yHat = lasso.predict(X_test)

    r2_scores["Lasso"] = r2(Y_test, yHat)
    explained_variance["Lasso"] = evs(Y_test, yHat)


def bayes(X_train, X_test, Y_train, Y_test):

    bayesian = BayesianRidge()
    bayesian.fit(X_train, Y_train)
    yHat = bayesian.predict(X_test)

    r2_scores["Bayesian"] = r2(Y_test, yHat)
    explained_variance["Bayesian"] = evs(Y_test, yHat)

def elasticNet(X_train, X_test, Y_train, Y_test):

    en = ElasticNet(alpha = 0.01)
    en.fit(X_train, Y_train)
    yHat = en.predict(X_test)

    r2_scores["ElasticNet"] = r2(Y_test, yHat)
    explained_variance["ElasticNet"] = evs(Y_test, yHat)

def model(data):
    categories = []
    for col in data.columns:
        if (col != 'SalePrice'):
            categories.append(col)
    X = data[categories].values
    Y = data['SalePrice'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    ols(X_train, X_test, Y_train, Y_test)
    ridge(X_train, X_test, Y_train, Y_test)
    lasso(X_train, X_test, Y_train, Y_test)
    bayes(X_train, X_test, Y_train, Y_test)
    elasticNet(X_train, X_test, Y_train, Y_test)

def outputResults():
    print("R-squared results:")
    for key in r2_scores:
        print("     " + key + ": " + str(r2_scores[key]))
    print("Explained variance results:")
    for key in explained_variance:
        print("     " + key + ": " + str(explained_variance[key]))

    # Determine highest r2 score
    max_r2 = "OLS"
    for key in r2_scores:
        if r2_scores[key] > r2_scores[max_r2]:
            max_r2 = key
    print("HIGHEST R2 SCORE: " + max_r2.upper())

def main():
    data = prepData()
    # plot(data)
    model(data)
    outputResults()

main()