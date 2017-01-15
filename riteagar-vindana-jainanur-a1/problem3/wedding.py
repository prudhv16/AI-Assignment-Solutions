#Author :- Prudhvi Indana
#AI Assignment 1 - problem 3


#Importing required modules
import sys
#from collections import OrderedDict
#from collections import deque
import copy



def tableoptimize(tables,lookupq):
    '''
    This recursive function finds the optimal way to arange people on wedding tables such that number of tables needed is less.
    This function searches through all the possible ways to arange people on tables recursively and returns the most optimal way.
    table :- initially an empty list.
    lookupq :- all the people attending the weeding arranged in assending order of friends they have.
    lookup :- dictioanry the is used to check if a person has a common friend during the method execution.
    Note :- lookup is a global variable and not explicitly passed to this function.
    '''
    while lookupq:
        currentfriend = lookupq.pop()
        tableoutput = [];
        appendswitch = True
        temptable = []
        for table in tables:
            if len(table) < tablelimit:
                peopleswitch = True
                for people in table:
                    #check if the friend can be added to the table by looking at lookup dictionary to determine if anyexisting person in tale is freind with
                    #current friend.
                    if currentfriend in lookup[people]:
                        peopleswitch = False
                #if currentfriend can be appending to current table recursively call tableoptimize and find solution for the subproblem.
                if peopleswitch == True:
                    table.append(currentfriend)
                    #take a new copy of tables, lookupq and recursively call tableoptimize to find a optimal solution for the subproblem.
                    temptables = copy.deepcopy(tables)
                    tempqueue = copy.deepcopy(lookupq)
                    temptable.append(tableoptimize(temptables,tempqueue))
                    table.pop()
                    appendswitch = False
        #if currentfriend can not be added to any table add him to a new table,
            
        if appendswitch == True:
            tables.append([currentfriend])
        else:
        #This returns the optimal table arangement to all the arangements possible at for a given table and lookupq combination.
            temptables = copy.deepcopy(tables)
            tempqueue = copy.deepcopy(lookupq)
            temptables.append([currentfriend])
            temptable.append(tableoptimize(temptables,tempqueue))
            return min(temptable, key=len)
    else:
        return tables

#greedy aproach
##for item,friends in reversed(lookup.items()):
##    appendswitch = True
##    for table in tables:
##        if len(table) < tablelimit:
##            peopleswitch = True
##            for people in table:
##                #print lookup[people]
##                if item in lookup[people]:
##                    peopleswitch = False
##            if peopleswitch == True:
##                table.append(item)
##                appendswitch = False
##                break
##    if appendswitch == True:
##        tables.append([item])

if __name__ == "__main__":
    inputfile,tablelimit = sys.argv[1:];tablelimit = int(tablelimit)
    file_contents = [line.strip("\n").split(" ") for line in open(inputfile, "r")]
    lookup = {item[0] : item[1:] for item in file_contents}
    tables = []
    for item,friends in lookup.items():
        for friend in friends:
            if friend in lookup:
                if item not in lookup[friend]:
                    lookup[friend].append(item)
            else:
                lookup[friend] = [item]
    ordereditem =  sorted(lookup.items(), key=lambda t: len(t[1]))
    '''because of python version problem I removed using collections deque and OrderedDict'''
    #print tempitem
    #lookup = OrderedDict(sorted(lookup.items(), key=lambda t: len(t[1])))
    #print 
    #print lookup
    #lookupq = deque()
    lookupq = []
    
    #for key,value in lookup.items():
    for key,value in ordereditem:
        lookupq.append(key)

    tables = tableoptimize(tables,lookupq)
    print len(tables),
    for table in tables:
        print ",".join(table),







