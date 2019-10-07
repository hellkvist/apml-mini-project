import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import truncnorm, norm, multivariate_normal as mvn
import time
import pandas as pd

COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
np.random.seed(1)
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
        (s_1[k], s_2[k]) = pdf_s_given_t_rvs(m1, m2, s1, s2, st, t[k-1])
        t[k] = pdf_t_given_s_and_y_rvs(s_1[k], s_2[k], st, y0)

    return s_1, s_2, t, time.clock() - timer_0

def run_q4():
    
    print('Running Gibbs Sampler for single match')
    
    burn_in = 50
    m1 = 25
    m2 = 25 
    s1 = 25/3
    s2 = 25/3
    st = 25/3 # small 'st' implies that the outcome is more telling of the skill
    y = 1 # i.e. player 1 wins
    t0 = 100
    K = 100
    print('K=',K)
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])

    # Plot the histogram of the samples generated (after
    # burn-in) together with the fitted Gaussian posterior for at least four (4) different numbers of
    # samples and report the time required to draw the samples.
    x_eval = np.linspace(10,50,100)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.figure(1, figsize=[4,3.5])
    ax = plt.subplot(221)
    plt.title('K='+ str(K-burn_in)+', time: '+str(np.round(runtime, 3))+'s', fontsize=8)
    plt.hist(s_1[burn_in::], color='C0', density=1,alpha=0.6, bins=50)
    plt.plot(x_eval, s_1_post.pdf(x_eval), 'C0')
    plt.ylim(0,.1)
    plt.xlim(10, 50)
    plt.xlabel(r'$s_1$', fontsize=9)
    plt.ylabel(r'$p(s_1|y=1)$', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)

    K = 200
    print('K=',K)
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
    plt.figure(1)
    ax = plt.subplot(222)
    plt.title('K='+ str(K-burn_in)+', time: '+str(np.round(runtime, 3))+'s', fontsize=8)
    plt.hist(s_1[burn_in::], color='C0', density=1,alpha=0.6, bins=50)
    plt.plot(x_eval, s_1_post.pdf(x_eval), 'C0')    
    plt.ylim(0,.1)
    plt.xlim(10, 50)
    plt.xlabel(r'$s_1$', fontsize=9)
    plt.ylabel(r'$p(s_1|y=1)$', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)

    K = 500
    print('K=',K)
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
    plt.xlabel(r'$s_1$', fontsize=9)
    plt.ylabel(r'$p(s_1|y=1)$', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)

    K = 1000
    print('K=',K)
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
    plt.figure(1)
    ax = plt.subplot(224)
    plt.title('K='+ str(K-burn_in)+', time: '+str(np.round(runtime, 3))+'s', fontsize=8)
    plt.hist(s_1[burn_in::], color='C0', density=1,alpha=0.6, bins=50)
    plt.plot(x_eval, s_1_post.pdf(x_eval), 'C0') 
    plt.ylim(0,.1)
    plt.xlim(10, 50)
    plt.xlabel(r'$s_1$', fontsize=9)
    plt.ylabel(r'$p(s_1|y=1)$', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94,
                        bottom=0.105,
                        left=0.165,
                        right=0.95,
                        hspace=0.5,
                        wspace=0.6)

    K = 500
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
        
    plt.figure(2, figsize=[3.5, 2.5])
    ax = plt.subplot(111)
    plt.subplots_adjust(top=0.96,
                        bottom=0.145,
                        left=0.16,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    plt.plot(s_1, label='s1')
    plt.plot(s_2, label='s2')
    plt.plot(t, label='t')
    plt.ylabel('Samples', fontsize=8)
    plt.xlabel('Gibbs iteration', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.legend(prop=fontP)
    plt.xlim(0,100)
    plt.grid()


    K = 500
    s_1, s_2, t, runtime = gibbs_q4(K, m1, m2, s1, s2, st, y, t0, burn_in)
    s_1_post, s_2_post = posterior(s_1[burn_in::], s_2[burn_in::])
        
    # Compare the prior p(s1) with the Gaussian approximation of the posterior p(s1|y = 1);
    # similarly compare p(s2) with p(s2|y = 1).
    s_1_prior = norm.freeze(loc=m1, scale=s1)
    s_2_prior = norm.freeze(loc=m2, scale=s2)

    x_range = np.arange(-20,60)
    # plt.figure(3)
    # ax = plt.subplot(111)
    # plt.plot(x_range, s_1_prior.pdf(x_range), 'C0', label='Prior') 
    # plt.plot(x_range, s_1_post.pdf(x_range), 'C1--', label='Posterior s1') 
    # plt.plot(x_range, s_2_post.pdf(x_range), 'C2--', label='Posterior s2') 
    # plt.legend(prop=fontP)
    # ax.tick_params(axis='both', which='major', labelsize=8)
    # plt.xlabel(r'$s_1, s_2$', fontsize=9)
    # plt.ylabel(r'$p(s_1|y=1), p(s_2|y=1)$', fontsize=9)
    # plt.grid()
    # plt.tight_layout()
    print('\n######################################\n')
    
def gibbs_q5(K, m1, m2, s1, s2, st, y0, t0, burn_in):
    # std inputs s1 s2 st
    s_1 = np.zeros(K)
    s_2 = np.zeros(K)
    t = np.zeros(K)
    t[0] = t0
    for k in range(K):
        if k == burn_in:
            timer_0 = time.clock()
        # if k % 1000 == 0:
        #     print(k, '/', K)
        (s_1[k], s_2[k]) = pdf_s_given_t_rvs(m1, m2, s1, s2, st, t[k-1]) # std inputs
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

    final_df = pd.DataFrame.from_dict(final_dict, orient='index', columns=['Skill mean', 'Skill std', 'Skill estimate'])
    print(final_df.sort_values(by=['Skill estimate'], ascending=0))

def run_q6(ordered):
    '''
        if ordered = 'True' or similar, the processing is done in chronological order
        if ordered = 'False', its the reversed.
    '''
    if ordered == True:
        print('Running Gibbs Sampler for Serie A, chronologically')
    elif ordered == False:
        print('Running Gibbs Sampler for Serie A, reverse chronologically')
    
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

    # standard deviation st for t:
    st = 25/3

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
                conservative_prediction_correct.append(False)
                prediction_correct.append(False)

            # update the dictionary for the two teams
            team_dict[t1][0].append(m1_new)
            team_dict[t1][1].append(s1_new)
            team_dict[t2][0].append(m2_new)
            team_dict[t2][1].append(s2_new)

    correct_prediction_rate = np.mean(prediction_correct)
    correct_conservative_prediction_rate = np.mean(conservative_prediction_correct)
    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    plt.title('Gibbs Sampler Serie A, Mean Skill Value')
    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    plt.title('Gibbs Sampler Serie A, TrueSkill Value')
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

    final_df = pd.DataFrame.from_dict(final_dict, orient='index', columns=['Skill mean', 'Skill std', 'Skill estimate'])
    print(final_df.sort_values(by=['Skill estimate'], ascending=0))
    print('\nprediction r: ', correct_prediction_rate)
    print('convseravative prediction r: ', correct_conservative_prediction_rate)
    print('\n######################################\n')

def gaussMul(mu1, s1, mu2, s2):
    '''
        mu1, mu2 are means
        s1, s2 are variances
    '''
    s = s1*s2/(s1 + s2)
    mu = (mu1*s2 + mu2*s1)/(s1 + s2)
    return mu, s

def gaussDiv(mu1, s1, mu2, s2):
    '''
        mu1, mu2 are means
        s1, s2 are variances
    '''
    s = s1*s2/(s2 - s1)
    mu = (mu1*s2 - mu2*s1)/(s2 - s1)
    return mu, s

def momentMatchTruncGauss(low, high, mu1, s1):
    '''
        low, high are the lower and upper bounds of truncation
        mu1, s1 are the mean and variance of the gaussian to truncate
    '''
    a = (low - mu1)/s1**0.5
    b = (high - mu1)/s1**0.5
    mu = truncnorm.mean(a, b, loc=mu1, scale=s1**0.5)
    s = truncnorm.var(a, b, loc=0, scale=s1**0.5)
    return mu, s

def message_passing(y_in, m1_p, s1_p, m2_p, s2_p, st_p):
    '''
        inputs s1_p, s2_p, st_p are prior VARIANCES for the RVs
        returns posterior means and standard deviations for players 1 and 2
    '''
    # message from priors to nodes s_1 and s_2
    mu_s1_m = m1_p
    mu_s1_s = s1_p
    mu_s2_m = m2_p
    mu_s2_s = s2_p

    # message from nodes s_1 and s_2 to factor f_sw
    mu_s1_w_m = mu_s1_m
    mu_s1_w_s = mu_s1_s
    mu_s2_w_m = mu_s2_m
    mu_s2_w_s = mu_s2_s

    # message from factor f_sw to node w
    mu_sw_w_m = mu_s1_w_m - mu_s2_w_m
    mu_sw_w_s = mu_s1_w_s + mu_s2_w_s

    # message from node w to factor f_wt
    mu_w_wt_m = mu_sw_w_m
    mu_w_wt_s = mu_sw_w_s

    # message from factor f_wt to node t
    mu_wt_t_m = mu_w_wt_m
    mu_wt_t_s = mu_w_wt_s + st_p

    # moment match to find marginal in node t
    if y_in == 1: # remove pdf below zero
        low = 0
        high = 10000
    elif y_in == -1: # remove pdf above zero
        low = -10000
        high = 0
    t_mm_mean, t_mm_s = momentMatchTruncGauss(low, high, mu_wt_t_m, mu_wt_t_s)

    # divide messages incoming to node t: moment matched from factor ty to t, and message from factor wt to t
    mu_hat_ty_t_m, mu_hat_ty_t_s = gaussDiv(t_mm_mean, t_mm_s, mu_wt_t_m, mu_wt_t_s)

    # message from node t to factor wt
    mu_hat_t_wt_m = mu_hat_ty_t_m
    mu_hat_t_wt_s = mu_hat_ty_t_s

    # message from factor wt to node w
    mu_hat_wt_t_m = mu_hat_t_wt_m
    mu_hat_wt_t_s = mu_hat_t_wt_s + st_p # we add the prior variance on t

    # message from node w to factor sw
    mu_hat_w_sw_m = mu_hat_wt_t_m
    mu_hat_w_sw_s = mu_hat_wt_t_s

    # message from factor sw to node s_1
    mu_hat_sw_s1_m = mu_s2_m + mu_hat_w_sw_m
    mu_hat_sw_s1_s = mu_s2_s + mu_hat_w_sw_s 

    # message from factor sw to node s_2
    mu_hat_sw_s2_m = mu_s1_m - mu_hat_w_sw_m
    mu_hat_sw_s2_s = mu_s1_s + mu_hat_w_sw_s

    # compute posteriors as products of ingoing prior and message from the factor sw
    p_s1_post_m, p_s1_post_s = gaussMul(mu_s1_m, mu_s1_s, mu_hat_sw_s1_m, mu_hat_sw_s1_s)
    p_s2_post_m, p_s2_post_s = gaussMul(mu_s2_m, mu_s2_s, mu_hat_sw_s2_m, mu_hat_sw_s2_s)

    # return posterior means and standard deviations
    return p_s1_post_m, p_s1_post_s**0.5, p_s2_post_m, p_s2_post_s**0.5

def run_q8_compare():
    print('Running comparison on Gibbs and MP posteriors')
    # means and std for players s_1, s_2 and variable t
    m1 = 25
    m2 = 25
    s1 = 25/3
    s2 = 25/3
    st = 4
    K = 500
    burn_in = 50
    y = 1 # player 1 wins

    m1_post, s1_post, m2_post, s2_post = message_passing(y, m1, s1**2, m2, s2**2, st**2) # takes variance as input
    skill_1, skill_2, _, _ = gibbs_q4(K, m1, m2, s1, s2, st, y, 10, burn_in) # takes std as input
    m1_post_g, s1_post_g = np.mean(skill_1[burn_in::]), np.std(skill_1[burn_in::])
    m2_post_g, s2_post_g = np.mean(skill_2[burn_in::]), np.std(skill_2[burn_in::])


    x = np.linspace(-10, 60, 10000)
    plt.figure(figsize=[3.5, 2.5])
    ax = plt.subplot(111)
    plt.subplots_adjust(top=0.96,
                        bottom=0.145,
                        left=0.16,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    plt.plot(x, norm.pdf(x, m1_post, s1_post), 'C1', label='Winner, MP')
    plt.plot(x, norm.pdf(x, m1_post_g, s1_post_g), 'C1--', label='Winner, Gibbs')
    plt.plot(x, norm.pdf(x, m2_post, s2_post), 'C2', label='Loser, MP')
    plt.plot(x, norm.pdf(x, m2_post_g, s2_post_g), 'C2--', label='Loser, Gibbs')
    plt.plot(x, norm.pdf(x, m1, s1), 'C0', label='Prior players')
    # plt.plot(x, norm.pdf(x, m2, s2), 'C0', label='Prior players')
    ax.tick_params(axis='both', which='major', labelsize=8)
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.xlabel(r'$s_1, s_2$', fontsize=8)
    plt.ylabel(r'$p(s_1|y=1),~p(s_2|y=1)$', fontsize=8)
    plt.legend(prop=fontP) 
    plt.grid()
    print('Finished! \n#######################################\n')

def run_mp_serie_A(ordered=True):
    '''
        if ordered = 'True' or similar, the processing is done in chronological order
        if ordered = 'False', its the reversed.
    '''
    print('Running Message Passing for SerieA')

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
    st = 4**2
    # st = 5
    # burn in from q4
    burn_in = 50
    K = 500

    if ordered == 1:
        range_play = range(M)
    elif ordered == 0:
        range_play = range(M-1, 0, -1)
    
    draws = 0
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
                # input outcome, means and variances
                m1_new, s1_new, m2_new, s2_new = message_passing(y[m], m1, s1**2, m2, s2**2, st)

                # prediction: t has mean s_1 - s_2 so we define the prediction as y = sgn(s_1 - s_2)
                y_prediction = np.sign(m1 - m2)
                y_cons_pred = np.sign((m1 - 3*s1) - (m2 - 3*s2))
                y_correct = np.sign(t[m])
                
                prediction.append(y_prediction)
                prediction_correct.append(y_correct == y_prediction)
                conservative_prediction.append(y_cons_pred)
                conservative_prediction_correct.append(y_correct == y_cons_pred)
            else:
                draws += 1
                m1_new = m1
                s1_new = s1
                m2_new = m2
                s2_new = s2
                prediction_correct.append(False)
                conservative_prediction_correct.append(False)
            # update the dictionary for the two teams
            team_dict[t1][0].append(m1_new)
            team_dict[t1][1].append(s1_new)
            team_dict[t2][0].append(m2_new)
            team_dict[t2][1].append(s2_new)

    correct_prediction_rate = np.mean(prediction_correct)
    correct_conservative_prediction_rate = np.mean(conservative_prediction_correct)
    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    plt.title('Message Passing for Serie A, mean skill values')
    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    plt.title('Message Passing for Serie A, TrueSkill values')
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
    print('no draws: ', draws)
    print('Finished! \n######################################\n')

def run_mp_serie_A_pred_draws(thresh, ordered=True):
    '''
        if ordered = 'True' or similar, the processing is done in chronological order
        if ordered = 'False', its the reversed.
    '''
    # print('Running Message Passing for SerieA with draw predictions')

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

    # for predictions with conservative skill estimate
    conservative_prediction = []
    conservative_prediction_correct = []

    # variance for t st:
    st = 4**2
    # st = 5
    # burn in from q4
    burn_in = 50
    K = 500

    if ordered == 1:
        range_play = range(M)
    elif ordered == 0:
        range_play = range(M-1, 0, -1)
    
    draws = 0
    for m in range_play:
    # for m in range(80):
            # if m % 20 == 0:
            #     print(m, '/', M)
            # determine what teams are playing
            t1 = team1[m]
            t2 = team2[m]

            # extract means and std for both teams
            m1 = team_dict[t1][0][-1]
            s1 = team_dict[t1][1][-1]
            m2 = team_dict[t2][0][-1]
            s2 = team_dict[t2][1][-1]

            if t[m] != 0: # we skip matches that were draw and repeat the previous value instead
                # input outcome, means and variances
                m1_new, s1_new, m2_new, s2_new = message_passing(y[m], m1, s1**2, m2, s2**2, st)
            else:
                draws += 1
                m1_new = m1
                s1_new = s1
                m2_new = m2
                s2_new = s2
            
            diff = (m1 - 3*s1) - (m2 - 3*s2) # prior values
            if diff > thresh/2: # home team wins
                y_pred = 1
            elif diff < -thresh: # away team wins
                y_pred = -1
            else: # draw
                y_pred = 0
            y_correct = np.sign(t[m])
            conservative_prediction.append(y_pred)
            conservative_prediction_correct.append(y_correct == y_pred)
            # update the dictionary for the two teams
            team_dict[t1][0].append(m1_new)
            team_dict[t1][1].append(s1_new)
            team_dict[t2][0].append(m2_new)
            team_dict[t2][1].append(s2_new)

    correct_conservative_prediction_rate = np.mean(conservative_prediction_correct)
    # fig1 = plt.figure()
    # ax1 = plt.subplot(111)
    # plt.title('Message Passing for Serie A, mean skill values')
    # fig2 = plt.figure()
    # ax2 = plt.subplot(111)
    # plt.title('Message Passing for Serie A, TrueSkill values')
    # c_count = 0
    # final_dict = {}
    # for team in team_names:
    #     m = np.array(team_dict[team][0])
    #     s = np.array(team_dict[team][1])
    #     m_fin = team_dict[team][0][-1]
    #     s_fin = team_dict[team][1 ][-1]
    #     final_dict[team] = list([np.round(m_fin, 2),np.round(s_fin, 2), np.round(m_fin-3*s_fin, 2)])
    #     # m_up = m + 0.1*s
    #     # m_dn = m - 0.1*s
    #     ax1.plot(m, color=COLORS[c_count], label=team)
    #     ax2.plot(m-3*s, color=COLORS[c_count], label=team)
    #     # plt.fill_between(x=np.arange(len(m)),y1=m_up, y2=m_dn, color=COLORS[c_count], alpha=0.5)
    #     c_count += 1
    # fontP = FontProperties()
    # fontP.set_size('xx-small')
    # plt.legend(prop=fontP, ncol=7) 
    # plt.grid()
    # plt.tight_layout()

    # final_df = pd.DataFrame.from_dict(final_dict, orient='index', columns=['Skill mean', 'Skill variance', 'Skill estimate'])
    # print(final_df.sort_values(by=['Skill estimate'], ascending=0))
    # print('convseravative prediction r: ', correct_conservative_prediction_rate)
    # print('no draws: ', draws)
    # print('Finished! \n######################################\n')
    return correct_conservative_prediction_rate

def run_mp_serie_A_tuner(st, ordered=True):
    '''
        st is variance for game result t

        if ordered = 'True' or similar, the processing is done in chronological order
        if ordered = 'False', its the reversed.
    '''
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

    # burn in from q4
    burn_in = 50
    K = 500

    if ordered == 1:
        range_play = range(M)
    elif ordered == 0:
        range_play = range(M-1, 0, -1)
    
    for m in range_play:
    # for m in range(80):
            # if m % 20 == 0:
            #     print(m, '/', M)
            # determine what teams are playing
            t1 = team1[m]
            t2 = team2[m]

            # extract means and std for both teams
            m1 = team_dict[t1][0][-1]
            s1 = team_dict[t1][1][-1]
            m2 = team_dict[t2][0][-1]
            s2 = team_dict[t2][1][-1]

            if t[m] != 0: # we skip matches that were draw and repeat the previous value instead
                # input outcome, means and variances
                m1_new, s1_new, m2_new, s2_new = message_passing(y[m], m1, s1**2, m2, s2**2, st)

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

    return correct_conservative_prediction_rate

def run_tune_st_mp_serieA():
    print('Running tuner over sigma_t.\nShowing progress:')
    N = 20000
    std_grid = np.linspace(0.1, 5, num=N)

    r = []
    for c, std in enumerate(std_grid,0):
        if c % 100 == 0:
            print(c, '/', N, end='\r')
        r.append(run_mp_serie_A_tuner(std**2))
    plt.figure()
    plt.plot(std_grid**2, r)
    plt.xlabel(r'$\sigma_t^2$')
    plt.ylabel('Correct Prediction Rate')
    plt.grid()
    plt.figure()
    plt.plot(std_grid, r)
    plt.xlabel(r'$\sigma_t$')
    plt.ylabel('Correct Prediction Rate')
    plt.grid()
    print('Finished!\n################################\n')

def run_tune_st_mp_serieA_thresh():
    print('Running tuner over sigma_t.\nShowing progress:')
    N = 200
    thresh_grid = np.linspace(0.01, 5, num=N)

    r = []
    for c, tr in enumerate(thresh_grid,0):
        if c % 10 == 0:
            print(c, '/', N, end='\r')
        r.append(run_mp_serie_A_pred_draws(tr))
    plt.figure()
    plt.plot(thresh_grid, r)
    plt.xlabel(r'Threshold')
    plt.ylabel('Correct Prediction Rate')
    plt.grid()
    print('Finished!\n################################\n')

def run_gibbs_shl(ordered=True):
    shl = pd.read_csv('shl_1.csv', sep=',')
    team_names = list(shl.team1.unique())
    N_teams = len(team_names)

    t = np.array(shl.score1 - shl.score2)
    y = np.sign(t)
    M = len(y) # number of matches
    
    team_dict = {}
    for team in team_names:
        team_dict[team] = list([[25], [25/3]])
    
    team1 = np.array(shl.team1)
    team2 = np.array(shl.team2)
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

    final_df = pd.DataFrame.from_dict(final_dict, orient='index', columns=['Skill mean', 'Skill std', 'Skill estimate'])
    print(final_df.sort_values(by=['Skill estimate'], ascending=0))
    print('\nprediction r: ', correct_prediction_rate)
    print('convseravative prediction r: ', correct_conservative_prediction_rate)

def run_mp_shl(ordered=True):
    '''
        if ordered = 'True' or similar, the processing is done in chronological order
        if ordered = 'False', its the reversed.
    '''
    shl = pd.read_csv('shl_1.csv')
    team_names = list(shl.team1.unique())
    N_teams = len(team_names)

    t = np.array(shl.score1 - shl.score2)
    y = np.sign(t)
    M = len(y) # number of matches
    
    team_dict = {}
    for team in team_names:
        team_dict[team] = list([[25], [25/3]])
    
    team1 = np.array(shl.team1)
    team2 = np.array(shl.team2)

    # initialize array for predictions, one element per match saying either -1 or +1
    prediction = []
    prediction_correct = [] # will have a value 1 in elements where the match was correctly predicted

    # for predictions with conservative skill estimate
    conservative_prediction = []
    conservative_prediction_correct = []

    # variance for t st:
    st = (25/3)**2

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
                # input outcome, means and variances
                m1_new, s1_new, m2_new, s2_new = message_passing(y[m], m1, s1**2, m2, s2**2, st)

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
                prediction_correct.append(False)
                conservative_prediction_correct.append(False)
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

run_q4() # generate figure 2 and figure 3 from the report
run_q6(ordered=1) # generate left part of table 1 in the report
run_q6(ordered=0) # generate table 3 in the report appendix 
run_mp_serie_A() #  generate right part of table 1 in the report
run_q8_compare() # generate figre 5 from the report
# run_gibbs_shl() # is not used for the report
run_mp_shl() # generate table 2 in the report
run_tune_st_mp_serieA() # generate figure 6 in the report OBS takes a long time (20 minutes?)
run_tune_st_mp_serieA_thresh() # generate figure 7 in the report OBS takes like 5-10 minutes.

print('Plotting...')
plt.show()    
print('Bye!')
