#! usr/bin/python
'''
************************************************
Author:- Prudhvi Indana
date :- 10/08/2016
playing yahtzee
choosing best action based on current die states
Note:- we always have three dies
************************************************
'''

import sys

def diceStateScoreCalc(state):
    '''
    Calculates the score of the current state
    helper function get score.
    '''
    die1 = state[0];die2 = state[1];die3 = state[2]
    return 25 if die1 == die2 and die2 == die3 else die1+die2+die3

def expected1ScoreCalc(die1):
    return sum([25 if x == die1 and x == y else x+y+die1 for x in range(1,7) for y in range(1,7)])/36.0

def expected2ScoreCalc(die1,die2):
    return sum([25 if x == die1 and die1 == die2 else x+die1+die2 for x in range(1,7)])/6.0

def expectedScoreCalc():
    return sum([25 if x==y and y ==z else x+y+z for x in range(1,7) for y in range(1,7) for z in range(1,7)])/216.0

def chooseBestSuccessor(state):
    '''
    Given the current state return the best possible action you can take based on your current state of dice.
    '''
    currentScore = diceStateScoreCalc(state)
    expected1Score = [expected1ScoreCalc(die) for die in state]
    print expected1Score
    expected1Score,value = max(expected1Score),expected1Score.index(max(expected1Score))
    expected2Score,value1,value2 = sorted([[expected2ScoreCalc(state[i],state[j]),i,j] for i in range(3) for j in range(i+1,3)],key = lambda x:x[0],reverse=True)[0]
    dievalue = [i for i in range(3) if i != value1 and i!= value2][0]
    dievalue1,dievalue2 = [i for i in range(3) if i != value]
    #print dievalu1,dievalue2
    #print dievalue
    print [[expected2ScoreCalc(state[i],state[j]),i,j] for i in range(3) for j in range(i+1,3)]
    expected3Score = expectedScoreCalc()
    print currentScore,expected1Score,expected2Score, expected3Score
    if currentScore >= expected1Score and  currentScore >= expected2Score and currentScore >= expected3Score:
        return "Leave dice as it is"
    elif expected1Score >= expected2Score and expected1Score >= expected3Score:
        return "Re roll dice having {0},{1} value".format(state[dievalue1],state[dievalue2])
    elif expected2Score >= expected3Score:
        return "Re roll the die having {0} value".format(state[dievalue])
    else:
        return "Re roll all the dies"



if __name__ == "__main__":
    state = map(int,sys.argv[1:])
    #print state,score
    print chooseBestSuccessor(state)
