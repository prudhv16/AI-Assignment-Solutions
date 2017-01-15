#!/usr/bin/python
####################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
# vindana (Venkata Prudhvi Raj Indana)
# sidpagad (Siddhartha Pagadala)
# rdandona (Rohit Dandona)
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#

'''
Here we try to predict mountain ridge line by using the edge strength map that measures how strong the image gradient is at each point.

Using th edge strength map we try finding the ridge line in below diffrent methods

Transition probability matrix is calculated as of i to j, pij = (number of rows-(difference of i and j))/sqrt(i+j) and normalize pij for all i
High transition probability if similar else less if dissimilar

Emission probability is calculated as normalization of the image gradient vector in a given column.

1)Simple Bayes method

Best state(row) was returned for each column by considering the emission probabilities alone.

2)Using MCMC

We are sampling every state given all other remaining state values from the distribution p(s_t|s_t-1,s_t+1, w_t) proportional to p(s_t|s_t-1) p(s_t+1|s_t) p(w_t|s_t)

The above equation is derived considering current state depends on previous state and next state depends on current state.

In gibbs sampler the initial states are obtained from the best possible sequence of the states of simple bayes.

Assumptions:-
	- a point lies on ridge line is it has higher image gradient
	- Transition probability from state i to state j will be high if they are close to each other
	- mountains are assumed to be near the top of the picture so more weightage for rows near top of the picture.

3)Improving the predictions using human input of a point on ridge line.

Knowing coordinates point present of the

'''

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
import math

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return filtered_y**2

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( max(y-thickness/2, 0), min(y+thickness/2, image.size[1]-1 ) ):
            image.putpixel((x, t), color)
    return image

#returns probable row for given edge strength and also returns emission probabulity for each column
#Assigns more probability for high image gradient row in a given column
#It is calculated as normalization of the image gradient vector in a given column
def get_row_simple(edge_strength):
    prob_edge_strength =   array([(edge_strength[:,i]+1)/(sum(edge_strength[:,i])+edge_strength.shape[0]) for i in range(edge_strength.shape[1])]).T
    probable_row = prob_edge_strength.argmax(axis=0)

    return  probable_row, prob_edge_strength

#returns transition probability array for all combination of states
#High transition probability if similar else less if dissimilar
#Calculated as transition probability of i to j, pij = (number of rows-(difference of i and j))/sqrt(i+j) and normalize pij for all i
def get_transition_prob(rows):
    trans_prob = array([[(rows-abs(r-c))/math.sqrt(r+c) for c in range(1, rows + 1)] for r in range(1, rows + 1)])
    trans_prob = trans_prob/trans_prob.sum(axis=0)[:,newaxis]
    return trans_prob

#viterbi version of part 2
#returns best possible sequence
def viterbi_cal_v(prob_edge_strength,trans_prob, rows,columns):
    v_t = []
    v_t.append(-math.log([1.0/float(rows)]*prob_edge_strength.shape[0])-math.log(prob_edge_strength[:,0]))
    v_t_backTrack = []

    for col_ind,c in enumerate(range(1,columns)):
        v_t_r = []
        v_t_r_backTrack = []
        for r in range(0,rows):
            v_t_prev = [-math.log(trans_prob[row,r])-math.log((v_t[c-1][row]))-math.log(prob_edge_strength[r,c]) for row in range(0,rows)]
            #v_t_prev = [-math.log((trans_prob[row, r]))-math.log(((v_t[c - 1][row]))) for row in range(0, rows)]
            min_index = v_t_prev.index(min(v_t_prev))
            v_t_r_backTrack.append(min_index)
            #v_t_r.append(prob_edge_strength[r,c] * trans_prob[min_index,r] * v_t[c-1][min_index])
            v_t_r.append(min(v_t_prev))
        v_t.append(v_t_r)
        v_t_backTrack.append(v_t_r_backTrack)

    last_seq_max = v_t[-1].index(min(v_t[-1]))
    v_t_backTrack = array(v_t_backTrack).T

    seq = []
    seq.append(last_seq_max)
    for ind, x in enumerate(range(v_t_backTrack.shape[1])):
        #print("index", ind)
        #print(last_seq_max)
        #print("length",v_t_backTrack.shape)
        last_seq_max = v_t_backTrack[last_seq_max,-ind-1]
        seq.append(last_seq_max)
    #print(seq)
    #print len(seq)
    return list(reversed(seq))

#returns best sequence
#gibbs sampling implementation of mcmc hmm
#Iteratively updates sequence, sampling from distribution of a sequence at time t(column) given all other column values(rows)
#computes mcmc samples for both with and with out  human feedback
#for human feedback, arguments passed --> gt_row = None, gt_col = Negative value less than -2 (Eg -3)
#p(s_t|s_t-1,s_t+1, w_t) proportional to p(s_t|s_t-1) p(s_t+1|s_t) p(w_t|s_t)
def gibbs(prob_rows, trans_prob, prob_edge_strength, gt_row, gt_col):
    rows,cols = prob_edge_strength.shape
    prob_rows = list(prob_rows)
    if gt_row != None:
        seq = prob_rows[:gt_col]+[gt_row]+prob_rows[gt_col+1:]
    else:
        seq = prob_rows
    for iter in range(10):
        for c in range(1, cols-1):
            if c != gt_col:
                v_t = [trans_prob[seq[c-1]][row]*prob_edge_strength[row][c]*trans_prob[row][seq[c+1]]*((rows-row)**2) for row in range(0,rows)]
                seq[c] = v_t.index(max(v_t))
                if c+1 != gt_col:
                    v_t = [trans_prob[seq[c]][row] * prob_edge_strength[row][c+1]*((rows-row)**2) for row in range(0, rows)]
                    seq[c+1] = v_t.index(max(v_t))
                if c-1 != gt_col:
                    v_t = [trans_prob[row][seq[c]] * prob_edge_strength[row][c -1]*((rows-row)**2) for row in range(0, rows)]
                    seq[c - 1] = v_t.index(max(v_t))
            else:
                v_t = [trans_prob[seq[c]][row] * prob_edge_strength[row][c + 1] * ((rows - row) ** 2) for row in range(0, rows)]
                seq[c + 1] = v_t.index(max(v_t))

                v_t = [ prob_edge_strength[row][c - 1] * ((rows - row) ** 2) for row in range(0, rows)]
                seq[c - 1] = v_t.index(max(v_t))

    return seq

#Splitting images based on given human feedback
#run gibbs samples on that
def gibbs_human(prob_rows, trans_prob, prob_edge_strength,gt_row, gt_col):
    seq1 = gibbs(prob_rows[:gt_col+1],trans_prob,prob_edge_strength[:,:gt_col+1],None,-3)
    seq2= gibbs(prob_rows[gt_col:],trans_prob,prob_edge_strength[:,gt_col:],None,-3)
    #print prob_edge_strength.shape
    #print seq1
    #print seq2
    return seq1+seq2[1:]
# main program
#input arguments
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
#computes probable rows for given edge strength and emission probability
prob_rows , prob_edge_strength = get_row_simple(edge_strength)

imsave('edges.jpg', edge_strength)

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.

ridge_simple = prob_rows
x = 'simple_bayes.jpg'
imsave(x, draw_edge(input_image, ridge_simple, (255, 0, 0), 5))
trans_prob = get_transition_prob(edge_strength.shape[0])
ridge_hmm =gibbs(prob_rows,trans_prob,prob_edge_strength,None,-3)
# output answer
temp_image = Image.open(x)
y = 'hmm.jpg'

imsave(y, draw_edge(temp_image, ridge_hmm, (0, 0, 255), 5))

#ridge_hmm_human = gibbs(prob_rows,trans_prob,prob_edge_strength,int(gt_row),int(gt_col))
ridge_hmm_human = gibbs_human(prob_rows,trans_prob,prob_edge_strength,int(gt_row),int(gt_col))

temp_image = Image.open(y)
imsave(output_filename, draw_edge(temp_image, ridge_hmm_human, (0, 255, 0), 5))








