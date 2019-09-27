import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import truncnorm, norm, multivariate_normal as mvn
import time
import pandas as pd

COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

def truncnorm_local_rvs(peak, std, sign):
    if sign == 1:
        return truncnorm.rvs(-peak/std, np.inf, scale=std) + peak
    else:
        return truncnorm.rvs(-np.inf, -peak/std, scale=std) + peak


def pdf_s_given_t_rvs(m1, m2, s1, s2, st, t):
    """
        m1: mean of variable s_1
        m2: mean of variable s_2
        s1: std of variable s_1
        s2: std of variable s_2
        st: std of variable t
        t: given sample of variable t
    """
    cov_s_given_t = np.linalg.inv( 1/(s1**2 * s2**2) * np.diag([s2**2, s1**2]) + 1/st**2 * np.array([[1, -1],[-1, 1]]) )
    cov_s_inv = np.linalg.inv(np.diag([s1**2, s2**2]))
    m_s = np.array([[m1],[m2]])
    M_transpose = np.array([[1],[-1]])

    m_s_given_t = cov_s_given_t @ (cov_s_inv @ m_s + M_transpose/st**2 * t)

    return mvn.rvs(mean=m_s_given_t.flatten(), cov=cov_s_given_t)   


def pdf_t_given_s_and_y_rvs(s_1, s_2, st, y):
    return truncnorm_local_rvs(s_1 - s_2, st, y)

def posterior(s_1_gibbs, s_2_gibbs):
    """
        given samples from gibbs sampler, returns frozen gaussians
    """
    return norm.freeze(loc=np.mean(s_1_gibbs), scale=np.std(s_1_gibbs)), norm.freeze(loc=np.mean(s_2_gibbs), scale=np.std(s_2_gibbs))

def gibbs_q4(K, m1, m2, s1, s2, st, y0, t0, burn_in):
    
    s_1 = np.zeros(K+1)
    s_2 = np.zeros(K+1)
    t = np.zeros(K+1)
    t[0] = t0
    for k in range(1, K+1):
        if k == burn_in:
            timer_0 = time.clock()
        if k % 1000 == 0:
            print(k, '/', K)
        (s_1[k], s_2[k]) = pdf_s_given_t_rvs(m1, m2, s1, s2, st, t[k-1])
        t[k] = pdf_t_given_s_and_y_rvs(s_1[k], s_2[k], st, y0)

    return s_1, s_2, t, time.clock() - timer_0

def run_q4():

    burn_in = 50
    m1 = 25
    m2 = 25 
    s1 = (25/3 )**.5
    s2 = (25/3)**.5
    st = (25/3)**.5 # small 'st' implies that the outcome is more telling of the skill
    y = 1 # i.e. player 1 wins
    t0 = 1000

    K = 1000
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])

    # Plot the histogram of the samples generated (after
    # burn-in) together with the fitted Gaussian posterior for at least four (4) different numbers of
    # samples and report the time required to draw the samples.
    x_eval = np.linspace(10,50,100)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.figure(1)
    ax = plt.subplot(221)
    plt.title('K='+ str(K-burn_in)+', time: '+str(np.round(runtime, 3))+'s', fontsize=8)
    plt.hist(s_1[burn_in::], color='C0', density=1,alpha=0.6, bins=50)
    plt.plot(x_eval, s_1_post.pdf(x_eval), 'C0')
    plt.ylim(0,.1)
    plt.xlim(10, 50)
    ax.tick_params(axis='both', which='major', labelsize=8)

    K = 200
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
    plt.figure(1)
    ax = plt.subplot(222)
    plt.title('K='+ str(K-burn_in)+', time: '+str(np.round(runtime, 3))+'s', fontsize=8)
    plt.hist(s_1[burn_in::], color='C0', density=1,alpha=0.6, bins=50)
    plt.plot(x_eval, s_1_post.pdf(x_eval), 'C0')    
    plt.ylim(0,.1)
    plt.xlim(10, 50)
    ax.tick_params(axis='both', which='major', labelsize=8)

    K = 500
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
    plt.figure(1)
    ax = plt.subplot(223)
    plt.title('K='+ str(K-burn_in)+', time: '+str(np.round(runtime, 3))+'s', fontsize=8)
    plt.hist(s_1[burn_in::], color='C0', density=1,alpha=0.6, bins=50)
    plt.plot(x_eval, s_1_post.pdf(x_eval), 'C0') 
    plt.ylim(0,.1)
    plt.xlim(10, 50)
    plt.xticks()
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    K = 1050
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
    plt.figure(1)
    ax = plt.subplot(224)
    plt.title('K='+ str(K-burn_in)+', time: '+str(np.round(runtime, 3))+'s', fontsize=8)
    plt.hist(s_1[burn_in::], color='C0', density=1,alpha=0.6, bins=50)
    plt.plot(x_eval, s_1_post.pdf(x_eval), 'C0') 
    plt.ylim(0,.1)
    plt.xlim(10, 50)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    
    K = 200
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
        
    plt.figure(2)
    ax = plt.subplot(111)
    plt.plot(s_1, label='s1')
    plt.plot(s_2, label='s2')
    plt.plot(t, label='t')
    plt.ylabel('Variable values', fontsize=8)
    plt.xlabel('Gibbs iteration', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.legend()
    plt.grid()


    K = 500
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
        
    # Compare the prior p(s1) with the Gaussian approximation of the posterior p(s1|y = 1);
    # similarly compare p(s2) with p(s2|y = 1).
    s_1_prior = norm.freeze(loc=m1, scale=s1)
    s_2_prior = norm.freeze(loc=m2, scale=s2)

    x_range = np.arange(-20,60)
    plt.figure(3)
    plt.plot(x_range, s_1_prior.pdf(x_range), 'C0', label='Prior') 
    plt.plot(x_range, s_1_post.pdf(x_range), 'C1--', label='Posterior s1') 
    plt.plot(x_range, s_2_post.pdf(x_range), 'C2--', label='Posterior s2') 
    plt.legend()
    plt.grid()
    plt.tight_layout()

    
def gibbs_q5(K, m1, m2, s1, s2, st, y0, t0, burn_in):

    s_1 = np.zeros(K)
    s_2 = np.zeros(K)
    t = np.zeros(K)
    t[0] = t0
    for k in range(K):
        if k == burn_in:
            timer_0 = time.clock()
        # if k % 1000 == 0:
        #     print(k, '/', K)
        (s_1[k], s_2[k]) = pdf_s_given_t_rvs(m1, m2, s1, s2, st, t[k-1])
        t[k] = pdf_t_given_s_and_y_rvs(s_1[k], s_2[k], st, y0)

    return s_1, s_2, t, time.clock() - timer_0

def run_q5(ordered):
    serie_A = pd.read_csv('SerieA.csv')
    team_names = list(serie_A.team1.unique())
    N_teams = len(team_names)

    t = np.array(serie_A.score1 - serie_A.score2)
    y = np.sign(t)
    M = len(y) # number of matches
    
    team_dict = {}
    for team in team_names:
        team_dict[team] = list([[25], [25/3]])
    
    team1 = np.array(serie_A.team1)
    team2 = np.array(serie_A.team2)


    # variance for t st:
    st = 5

    # burn in from q4
    burn_in = 50
    K = 500

    if ordered == 1:
        range_play = range(M)
    elif ordered == 0:
        range_play = range(M-1, 0, -1)
    
    for m in range_play:
            if m % 20 == 0:
                print(m, '/', M)
            # determine what teams are playing
            t1 = team1[m]
            t2 = team2[m]

            # extract means and std for both teams
            m1 = team_dict[t1][0][-1]
            s1 = team_dict[t1][1][-1]
            m2 = team_dict[t2][0][-1]
            s2 = team_dict[t2][1][-1]

            if t[m] != 0: # we skip matches that were draw and repeat the previous value instead
                # get samples from the gibbs sampler
                s_1, s_2, _, _ = gibbs_q5(K, m1, m2, s1, s2, st, y[m], t[m], burn_in)

                # compute mean and std disregarding the burn in phase
                m1_new = np.mean(s_1[burn_in::])
                s1_new = np.std(s_1[burn_in::])
                m2_new = np.mean(s_2[burn_in::])
                s2_new = np.std(s_2[burn_in::])
            else:
                m1_new = m1
                s1_new = s1
                m2_new = m2
                s2_new = s2
            # update the dictionary for the two teams
            team_dict[t1][0].append(m1_new)
            team_dict[t1][1].append(s1_new)
            team_dict[t2][0].append(m2_new)
            team_dict[t2][1].append(s2_new)

    plt.figure()
    c_count = 0
    final_dict = {}
    for team in team_names:
        m = np.array(team_dict[team][0])
        s = np.array(team_dict[team][1])
        m_fin = team_dict[team][0][-1]
        s_fin = team_dict[team][1 ][-1]
        final_dict[team] = list([np.round(m_fin, 2),np.round(s_fin, 2), np.round(m_fin-3*s_fin, 2)])
        # m_up = m + 0.1*s
        # m_dn = m - 0.1*s
        plt.plot(m, color=COLORS[c_count], label=team)
        # plt.fill_between(x=np.arange(len(m)),y1=m_up, y2=m_dn, color=COLORS[c_count], alpha=0.5)
        c_count += 1
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(prop=fontP, ncol=7) 
    plt.grid()
    plt.tight_layout()

    final_df = pd.DataFrame.from_dict(final_dict, orient='index', columns=['Skill mean', 'Skill variance', 'Skill estimate'])
    print(final_df.sort_values(by=['Skill estimate'], ascending=0))


def run_q6(ordered):
    serie_A = pd.read_csv('SerieA.csv')
    team_names = list(serie_A.team1.unique())
    N_teams = len(team_names)

    t = np.array(serie_A.score1 - serie_A.score2)
    y = np.sign(t)
    M = len(y) # number of matches
    
    team_dict = {}
    for team in team_names:
        team_dict[team] = list([[25], [25/3]])
    
    team1 = np.array(serie_A.team1)
    team2 = np.array(serie_A.team2)

    # initialize array for predictions, one element per match saying either -1 or +1
    prediction = []
    prediction_correct = [] # will have a value 1 in elements where the match was correctly predicted

    # for predictions with conservative skill estimate
    conservative_prediction = []
    conservative_prediction_correct = []

    # variance for t st:
    st = 5

    # burn in from q4
    burn_in = 50
    K = 500

    if ordered == 1:
        range_play = range(M)
    elif ordered == 0:
        range_play = range(M-1, 0, -1)
    
    for m in range_play:
    # for m in range(80):
            if m % 20 == 0:
                print(m, '/', M)
            # determine what teams are playing
            t1 = team1[m]
            t2 = team2[m]

            # extract means and std for both teams
            m1 = team_dict[t1][0][-1]
            s1 = team_dict[t1][1][-1]
            m2 = team_dict[t2][0][-1]
            s2 = team_dict[t2][1][-1]

            if t[m] != 0: # we skip matches that were draw and repeat the previous value instead
                # get samples from the gibbs sampler
                s_1, s_2, _, _ = gibbs_q5(K, m1, m2, s1, s2, st, y[m], t[m], burn_in)

                # compute mean and std disregarding the burn in phase
                m1_new = np.mean(s_1[burn_in::])
                s1_new = np.std(s_1[burn_in::])
                m2_new = np.mean(s_2[burn_in::])
                s2_new = np.std(s_2[burn_in::])

                # prediction: t has mean s_1 - s_2 so we define the prediction as y = sgn(s_1 - s_2)
                y_prediction = np.sign(m1 - m2)
                y_cons_pred = np.sign((m1 - 3*s1) - (m2 - 3*s2))
                y_correct = np.sign(t[m])
                
                prediction.append(y_prediction)
                prediction_correct.append(y_correct == y_prediction)
                conservative_prediction.append(y_cons_pred)
                conservative_prediction_correct.append(y_correct == y_cons_pred)
            else:
                m1_new = m1
                s1_new = s1
                m2_new = m2
                s2_new = s2
            # update the dictionary for the two teams
            team_dict[t1][0].append(m1_new)
            team_dict[t1][1].append(s1_new)
            team_dict[t2][0].append(m2_new)
            team_dict[t2][1].append(s2_new)

    correct_prediction_rate = np.mean(prediction_correct)
    correct_conservative_prediction_rate = np.mean(conservative_prediction_correct)
    fig1 = plt.figure(1)
    ax1 = plt.subplot(111)
    fig2 = plt.figure(2)
    ax2 = plt.subplot(111)
    c_count = 0
    final_dict = {}
    for team in team_names:
        m = np.array(team_dict[team][0])
        s = np.array(team_dict[team][1])
        m_fin = team_dict[team][0][-1]
        s_fin = team_dict[team][1 ][-1]
        final_dict[team] = list([np.round(m_fin, 2),np.round(s_fin, 2), np.round(m_fin-3*s_fin, 2)])
        # m_up = m + 0.1*s
        # m_dn = m - 0.1*s
        ax1.plot(m, color=COLORS[c_count], label=team)
        ax2.plot(m-3*s, color=COLORS[c_count], label=team)
        # plt.fill_between(x=np.arange(len(m)),y1=m_up, y2=m_dn, color=COLORS[c_count], alpha=0.5)
        c_count += 1
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(prop=fontP, ncol=7) 
    plt.grid()
    plt.tight_layout()

    final_df = pd.DataFrame.from_dict(final_dict, orient='index', columns=['Skill mean', 'Skill variance', 'Skill estimate'])
    print(final_df.sort_values(by=['Skill estimate'], ascending=0))
    print('\nprediction r: ', correct_prediction_rate)
    print('convseravative prediction r: ', correct_conservative_prediction_rate)


run_q4()
# run_q5(ordered=1)
# run_q5(ordered=0)
# run_q6(ordered=1)
plt.show()    

