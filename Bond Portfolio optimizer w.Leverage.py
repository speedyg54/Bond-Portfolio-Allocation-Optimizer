# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 19:53:59 2021

@author: Chris O'Bar

Goal: Create an optimizer using sequential least squares programming given below constraints

constraints:
    1) X - Asset Specific Min and Max weights
    2) X - duration range of 2 - 8
    3) X - Minimum Avg Bond rating of A+
    4) X - 20% of portfolio needs to have same day liqduidity	
        Liquidity Tier	Translation
        1	Same Day
        2	T+2
        3	T+5
    
    5) X - Sector Level constraints
            	Min	Max
            TSY	10%	100%
            ABS	0%	20%
            MBS	0%	40%
            Corp	0%	50%
            High Yield	0%	5%

"""
#Begin by importing necessary packages
import pandas as pd
import numpy as np
from scipy.optimize import minimize
#Import data
#Import data using read_excel, make sure the path is relevant to your environment...
df = pd.read_excel('C:\\Users\\<user>\\Documents\\Optimizer Project.xlsx', sheet_name='Data',
                   engine='openpyxl')


#Extract the Key Variables
xYields = np.array(df['Yield'].values.tolist())
xLBoundAsst = np.array(df['Asset Level Min Weight'].values.tolist())
xUBoundAsst = np.array(df['Asset Level Max Weight'].values.tolist())
xDuration = np.array(df['Duration'].values.tolist())
xSecurity = np.array(df['Asset'].values.tolist())
xBondQuality = np.array(df['Categorized Quality'].values.tolist())
xBondLiqRating = np.array(df['Liquidity Tier'].values.tolist())
xSector = pd.get_dummies(df['Sector '], prefix='Sector')
#array that holds initial set of weights 
xWeight = [0] * df.shape[0]
###Change the liquidity ratings to be 0 if theyre greater than 1
xBondLiqRating2 = np.where(xBondLiqRating == 1, 0, xBondLiqRating)
xBondLiqRating3 = np.where(xBondLiqRating2 > 1, 1, xBondLiqRating2)
###Sector Level Constraints, create separate arrays for each just like the liquidity ratings above
xSecTSY = np.array(xSector['Sector_TSY'].values.tolist())
xSecABS = np.array(xSector['Sector_ABS'].values.tolist())
xSecMBS = np.array(xSector['Sector_MBS'].values.tolist())
xSecCorp = np.array(xSector['Sector_Corp'].values.tolist())
xSecHY = np.array(xSector['Sector_High Yield'].values.tolist())

#build our optimizer
def optimizer(funct, Weights, Er, Duration, BondQualRating, BondLiqRating, LBound, UBound,
              SecTSY, SecABS, SecMBS, SecCorp, SecHY):
    final_Bounds = tuple(zip(LBound, UBound)) #Asset level min and max weights
    #build constraints
    final_constraints = (
                   # {'type' : 'eq', 'fun': lambda x: 1.0 - np.sum(x)}, #removed to include leverage.
                    {'type' : 'ineq', 'fun': lambda x, Duration: (x * Duration).sum() - 2, 'args': (Duration,)} #Duration >= 2
                    ,{'type' : 'ineq', 'fun': lambda x, Duration: 8-((x * Duration).sum()), 'args': (Duration,)} #Duration <= 8
                    ,{'type' : 'ineq', 'fun': lambda x, BondQualRating: 5-((x * BondQualRating).sum()), 'args': (BondQualRating,)} #BondqualRating <= 5
                    ,{'type' : 'ineq', 'fun': lambda x, BondLiqRating: .8-((x * BondLiqRating).sum()), 'args':(BondLiqRating,)} #20% of bond portfolio needs same day liquidity
                   #Sector Level Imposed Constraints
                    ###Treasury
                    ,{'type' : 'ineq', 'fun': lambda x, SecTSY: 1-((x * SecTSY).sum()), 'args':(SecTSY,)} #Tsy <= 100%
                    ,{'type' : 'ineq', 'fun': lambda x, SecTSY: ((x * SecTSY).sum()) - .1, 'args':(SecTSY,)} #Tsy >= 10%
                    ###Agency Backed Securities
                    ,{'type' : 'ineq', 'fun': lambda x, SecABS: .2-((x * SecABS).sum()), 'args':(SecABS,)} #ABS <= 20%
                    #,{'type' : 'ineq', 'fun': lambda x, SecABS: ((x * SecABS).sum()) - .1, 'args':(SecABS,)} #ABS >= 0%, uneccessary due to asset specific weights
                    ###Mortgage Backed Securities
                    ,{'type' : 'ineq', 'fun': lambda x, SecMBS: .4-((x * SecMBS).sum()), 'args':(SecMBS,)} #MBS <= 40%
                    #,{'type' : 'ineq', 'fun': lambda x, SecMBS: ((x * SecMBS).sum()) - .1, 'args':(SecMBS,)} #MBS >= 0%, uneccessary due to asset specific weights
                    ###Corporate Bonds
                    ,{'type' : 'ineq', 'fun': lambda x, SecCorp: .5-((x * SecCorp).sum()), 'args':(SecCorp,)} #Corp <= 50%
                    #,{'type' : 'ineq', 'fun': lambda x, SecCorp: ((x * SecCorp).sum()) - .1, 'args':(SecCorp,)} #Corp >= 0%, uneccessary due to asset specific weights
                    ###HY Bonds
                    ,{'type' : 'ineq', 'fun': lambda x, SecHY: .05-((x * SecHY).sum()), 'args':(SecHY,)} #HY <= 5%
                    #,{'type' : 'ineq', 'fun': lambda x, SecHY: ((x * SecHY).sum()) - .1, 'args':(SecHY,)} #HY >= 0%, uneccessary due to asset specific weights
                                        
                    # ,{'type' : 'ineq', 'fun': const2b(xDuration, 8)}
                        )   
    #Maximize Yield against weights where weights and other elements are constraints.
    final_weights = minimize(funct, Weights, args=(Er),
                             method= 'SLSQP',
                             bounds=final_Bounds,
                             constraints= final_constraints)

    
    
    return final_weights['x']
#Function to optimize
def ret_func(W, exp_ret):
    #minimize function acts as maximize when multiplying return by -1, also, W.T is tranposed weights, exp_ret are yields. @ is matrix multiplication
    return -(W.T@exp_ret)

x= optimizer(ret_func, xWeight, xYields, xDuration,xBondQuality, xBondLiqRating3, xLBoundAsst, xUBoundAsst
             ,xSecTSY, xSecABS, xSecMBS, xSecCorp, xSecHY)
#Round to .1 bps to make the final weights look nice.
x = np.round(x, decimals=5)

#Final portfolio:
print("Optimal Weights are: \n",
      np.array([[d, c] for d, c in zip(xSecurity, x)])
      )

#Total final portfolio utilized: 
print("Total Portfolio utilized: {0:.0%}".format(x.sum()))
#duration given optimal weights
finalduration = sum(x * y for x, y in zip(xDuration, x))
print("Final Portfolio Duration: {0:.5f}".format(finalduration))
#Final Average Bond Rating
finalbondrating = sum(x * y for x, y in zip(xBondQuality, x))
print("Final Avg. Bond Rating: {0:.5f}".format(finalbondrating)) #5 or less is A+
#Final portfolio yield
finalyield = sum(x * y for x, y in zip(xYields, x))
print("Final Portf. Yield: {0:.5f}".format(finalyield))
#Final Liquidity Rating
finaliquidityRating = 1-sum(x * y for x, y in zip(xBondLiqRating3, x))
print("Final Portfolio %'ge with Same day liquidity: {0:.0%}".format(finaliquidityRating)) #need same day liquidity of 20% in portfolio
