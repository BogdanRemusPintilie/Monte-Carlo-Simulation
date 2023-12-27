import numpy
import matplotlib.pyplot as plt
from pandas_datareader import data
import datetime
import yfinance
yfinance.pdr_override()

stockList = ['CBA', 'BHP'] #choose stocks
stocks = [stock + '.AX' for stock in stockList] #some conversion necessary to make the correct reference
endDate= datetime.datetime.now() # today
startDate= endDate - datetime.timedelta(days=300) #300 days ago

#let's import data

def get_data(stocks, start, end): # a function where you get all the stuff below
    stockData = data.get_data_yahoo(stocks, start, end)  #essentially a df with all Yahoo info for the stocks between the dates
    stockData = stockData["Close"] # I only care about the Close column = close price (essentially you get the close for each of the 300 days)
    returns = stockData.pct_change() # I believe the percentage day from one day to another for all days
    meanReturns = returns.mean() #mean percentage change for a stock between the dates spceified
    covMatrix = returns.cov() #covariance matrix of the percentage changes of the stocks betweeen the dates ## if you run print covMatrix I got no clue why ther isn't 1 on the diagonal
    return meanReturns, covMatrix #returns the values for those 2 not yet defined variables

meanReturns, covMatrix = get_data(stocks, startDate, endDate) # here you assign the values above to the now variables

weights = numpy.random.random(len(stockList)) #length = as many mean returns = as many stocks # number of stocks
weights /= numpy.sum(weights) #slick thing = normalizes the values of the weights = they add up to 1

#Monte Carlo Method
mc_sims = 100 #number of monte carlo simulations
T = 100 #time range (it is not the last T days)
initialPortfolio = 10000 # say you started with this on startDate

####################################

meanM = numpy.full(shape=(T, len(stockList)), fill_value=meanReturns) #create a matrix that contains the meanReturns                      #HERE I NEED MORE EXPLANATION: How do I know it will look at both stocks like its not a look ar anything
meanM = meanM.T #trnaspose of the matrix above                                                                                                                           # do you need the transpose like look at dailyReturns and what I said cause it comes from there

portfolio_sims = numpy.full(shape=(T, mc_sims), fill_value=0.0)

for m in range(0, mc_sims): #loop over the number of Monte Carlo simulations
    Z = numpy.random.normal(size=(T, len(stockList))) #Generates a matrix Z with random normal values (between -1 and 1), mimicking random stock movements over T days
    L = numpy.linalg.cholesky(covMatrix) #is used to transform the random normal values (Z) into CORRELATED random values to simulate correlated stock movements #essentially it chooses a correlation between the %changes at rendom and since in loop you do it mc_sims times
    dailyReturns= meanM + numpy.inner(L,Z) #supposed to be a matrix of daily returns (2,100) cause look at inner prod below                                                                                                                              # I don't think this is correct. I would either flip the order but you must do L first (obv AB!=BA) then numpy.dot(L,Z.T).T
    portfolio_sims[:,m] = numpy.cumprod(numpy.inner(weights, dailyReturns.T)+1)*initialPortfolio #Calculates the simulated portfolio value for each day in the simulation.                                   #Get some help with the +1 like in terms on math/linear algebra plus if you can see an illustratio that would also help
                       #for all the days T for the specific simulation             #HERE I NEED SOME HELP AS WELL

#######################

plt.plot(portfolio_sims) #plot the simulation
plt.ylabel("Portfolio Value $")
plt.xlabel("Days")
plt.title ("MC simulation of a stock portfolio")
plt.show()