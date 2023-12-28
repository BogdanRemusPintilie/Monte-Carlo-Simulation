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

stockData = data.get_data_yahoo(stocks, startDate, endDate)  #essentially a df with all Yahoo info for the stocks between the dates
stockData = stockData["Close"] # I only care about the Close column = close price (essentially you get the close for each of the 300 days)
returns = stockData.pct_change() # I believe the percentage day from one day to another for all days
meanReturns = returns.mean() #mean percentage change for a stock between the dates spceified
covMatrix = returns.cov() #all it tells is on average they tend to move in the same way or opposite ways   #if one is above it's average then the other is above it's average on average

#Monte Carlo Method
mc_sims = 100 #number of monte carlo simulations
T = 100 #the simulation is performed over a period of T days   #T has nothing to do with startDate

meanM = numpy.full(shape=(T, len(stockList)), fill_value=meanReturns) #create a matrix (100,2) that contains the meanReturns   #first column is meanreturn of 1st stock and second....


portfolio_sims = numpy.full(shape=(T, mc_sims), fill_value=0.0) #matrix (100,100) with 0s #essentially this is used to store all the portfolio_sims[:,m] so that when you plot you plot the performace of all simulations
weights = numpy.random.random(len(stockList)) # generates as many random numbers as stocks, the result is a 1D array != list. You can +1 and all numbers go up by 1 instead of getting an error cause you can't perform that operation with a list
weights /= numpy.sum(weights) #normalizes the random numbers
initialPortfolio = 10000 # say you started with this at the begining of T

for m in range(0, mc_sims): #loop over the number of Monte Carlo simulations
    Z = numpy.random.normal(size=(T, len(stockList))) #Generates a matrix Z with random normal values (you can input as arg mean=0, std=1, size(...)), mimicking random stock price changes over T days
    L = numpy.linalg.cholesky(covMatrix) #this is basically a matrix that multiplied by it's transpose gives you the covMatrix         #you use this to make the random numbers in the Z follow the same correlation structure as the one you can actually inferr from the covMatrix (for the stock returns)  #the random numbers now at least follow the same relationship as the returns of the stocks if you do the multiplication below
    dailyReturns= meanM + numpy.inner(Z,L.T)# add the mean return to all the random now correlated returns     #say the inner is 0, then the daily return is the mean so if you do it over time you would get to the same place as the actual real life performace.  To that now you add some randomness, could be for greater or worse outcome, could me for much or insignificant as in magnitude, don't know (i mean you can guess cause the random numbers are almost all within std=3 of the mean which is 0 so...)                  
    portfolio_sims[:,m] = numpy.cumprod(numpy.inner(weights, dailyReturns)+1)*initialPortfolio # you necesarily need the +1 there cause when you cumprod you will have 1.03* 1.05 and that will give the overall return of day 2 (imagine you go 0.03*0.05 and now ur overall return is lower than any of the returns incurred)  #plus you can go +1 cause the inner prod is an array so it will add 1 to all elements, if it was a list you coudn't have perfromed the operation
    #Calculates the simulated portfolio value for each day in the simulation. #the cumprod spits out the cumulative return so say if day 1 =+5%, day 2= +7% then the cummulative value for day 2 = 5%*7%, and further you go the more multiplications => in a (1,100) matrix to which if you * the initial value you get how much you were up or down by AT THE END of the respective day (like the change)# if you add the initial value then you get your overall standing AT THE END OF EACH DAY # the[:,m] essentially fills out the empty (100,100) matrix, so for all Ts and this particular simulation m it accesses the column in the portfolio_sims empty array and fills it with the result obtained from what you have on the right side of the equal. Do it for all m in mc_sims and you fill the portfolio_sims empty matrix entirely # like you want to plot all the simulations in one chart reason for the existance of the (100,100) empty matrix

plt.plot(portfolio_sims) #plot the result of the simulations over days # the plot tracks the variables which are the columns so in this case the simulations over T days so you know which axis is T and which is the variable, again the plot tracks the variable=columns aka column values
plt.ylabel("Portfolio Value $")
plt.xlabel("Days")
plt.title ("MC simulations of a stock portfolio")
plt.show()