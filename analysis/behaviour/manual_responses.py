import numpy as np
import os, sys
import os.path as op
import pandas as pd
import seaborn as sns
import yaml

from sklearn.linear_model import LinearRegression
import scipy

class BehResponses:
    
    """Behavioral Responses 
    Class that takes care of getting reaction times and accuracy for a given task
    """
    
    def __init__(self, dataObj):
        
        """__init__
        constructor for class, takes experiment params and subject num as input
        
        Parameters
        ----------
        dataObj : BehTask object
            object from one of the classes defined in behaviour.load_beh_data
            
        """
        
        # set data object to use later on
        self.dataObj = dataObj

        # set a boolean dict - will be used to track pp that we exclude due to performance
        self.exclude_sj_bool = {}
        for pp in self.dataObj.sj_num:
            self.exclude_sj_bool['sub-{sj}'.format(sj = pp)] = False
        
    
    def update_dfs(self, df, excl_list, col_name = 'sj'):

        """
        Helper function to update dataframe given excluded sj list
        
        """ 
        
        new_df = df[~df[col_name].isin(excl_list)]

        return new_df

    def get_RTs(self, missed_trl_thresh = .25, exclude_outliers = False, threshold_std = 3, threshold_sec = .250):
        
        """
        Given subject reponses and trial info, 
        returns data frame with reaction time for all trials
        
        """ 
        
        ####### if crowding task ###########
        if self.dataObj.task_name.lower() == 'crowding':
            
            ## all crowding types (includes uncrowding)
            self.crowding_type_all = self.dataObj.crwd_type + ['unflankered']
            
            # set up empty df
            df_manual_responses = pd.DataFrame({'sj': [], 'trial_num': [] ,'crowding_type': [],  'RT': [], 
                       'correct_response': [], 'correct_color': [], 'correct_ori': []})
            
            ## loop over subjects
            for pp in self.dataObj.sj_num:

                print('getting reaction times for crowding task of sub-{sj}'.format(sj = pp))

                ## loop over crowding types
                for crwd_type in self.crowding_type_all:
                    
                    # subselect trial info df to only have trials from this crowding type
                    df_crwd_type_info = self.dataObj.trial_info_df[(self.dataObj.trial_info_df['crowding_type'] == crwd_type) & \
                                                                    (self.dataObj.trial_info_df['sj'] == 'sub-{sj}'.format(sj = pp))]
                    
                    # trial numbers for this crowding type
                    trials_crwd_type = df_crwd_type_info['index'].values
                    
                    # loop over those trials
                    for t in trials_crwd_type:
                        
                        # events df for that trial
                        ev_df = self.dataObj.events_df[(self.dataObj.events_df['trial_nr'] == t) & \
                                                    (self.dataObj.events_df['sj'] == 'sub-{sj}'.format(sj = pp))] 

                        ## if participant responded
                        if 'response' in ev_df['event_type'].values:

                            # participant response key
                            response_key = ev_df[ev_df['event_type'] == 'response']['response'].values[0]

                            # trial target name
                            trl_targt = df_crwd_type_info[df_crwd_type_info['index'] == t]['target_name'].values[0]

                            ## if correct response
                            if response_key in self.dataObj.task_keys[trl_targt]:
                                correct_response = 1
                                correct_color = 1 
                                correct_ori = 1
                            ## if correct color but not orientation
                            elif response_key in [self.dataObj.task_keys[l][0] for l in self.dataObj.task_keys if trl_targt[0] in l]:
                                correct_response = 0
                                correct_color = 1 
                                correct_ori = 0
                            ## if correct orientation but not color
                            elif response_key in [self.dataObj.task_keys[l][0] for l in self.dataObj.task_keys if trl_targt[-1] in l]:
                                correct_response = 0
                                correct_color = 0
                                correct_ori = 1
                            ## simply incorrect
                            else:
                                correct_response = 0
                                correct_color = 0
                                correct_ori = 0

                            # save reaction time value
                            rt = ev_df[ev_df['event_type'] == 'response']['onset'].values[0] - ev_df[ev_df['event_type'] == 'stim']['onset'].values[0]

                            # save dataframe
                            df_manual_responses = pd.concat([df_manual_responses, 
                                                    pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                                'trial_num': [int(t)],
                                                                'crowding_type': [crwd_type],  
                                                                'RT': [rt], 
                                                                'correct_response': [correct_response], 
                                                                'correct_color': [correct_color], 
                                                                'correct_ori': [correct_ori]})
                                                ])

                        ## no responses given, missed trial
                        else: 
                            df_manual_responses = pd.concat([df_manual_responses, 
                                                    pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                                'trial_num': [int(t)],
                                                                'crowding_type': [crwd_type],  
                                                                'RT': [np.nan], 
                                                                'correct_response': [np.nan], 
                                                                'correct_color': [np.nan], 
                                                                'correct_ori': [np.nan]})
                                                ])

                ## if too many missed trials, then should exclude participant
                pp_trials_resp = df_manual_responses[df_manual_responses.sj == 'sub-{sj}'.format(sj = pp)].correct_response.values
                pp_trials_missed = np.sum(np.isnan(pp_trials_resp))/len(pp_trials_resp)

                print('missed %.2f %% of trials'%(pp_trials_missed*100))

                if  pp_trials_missed >= missed_trl_thresh:
                    print('EXCLUDE')
                    self.exclude_sj_bool['sub-{sj}'.format(sj = pp)] = True


        ####### if search task ########
        elif self.dataObj.task_name.lower() == 'visualsearch':
            
            # set up empty df
            df_manual_responses = pd.DataFrame({'sj': [], 'trial_num': [], 'block_num': [] ,'target_ecc': [], 'set_size': [], 
                                                'RT': [], 'correct_response': []})

            ## loop over subjects
            for pp in self.dataObj.sj_num:

                print('getting reaction times for search task of sub-{sj}'.format(sj = pp))

                # number of blocks in task (later participants had more blocks but same number of trials)
                nr_blocks = self.dataObj.trial_info_df.loc[self.dataObj.trial_info_df['sj'] == 'sub-{sj}'.format(sj = pp)]['block'].unique()

                print('nr of blocks %i'%len(nr_blocks))

                # block start onsets 
                blk_start_onsets = self.dataObj.events_df[(self.dataObj.events_df['event_type'] == 'block_start') & \
                                                            (self.dataObj.events_df['response'] == 'space') & \
                                                            (self.dataObj.events_df['sj'] == 'sub-{sj}'.format(sj = pp))]['onset'].values

                ## loop over blocks
                for blk in nr_blocks:

                    # subselect trial info df to only have trials from this block
                    df_blk_info = self.dataObj.trial_info_df[(self.dataObj.trial_info_df['block'] == blk) & \
                                                            (self.dataObj.trial_info_df['sj'] == 'sub-{sj}'.format(sj = pp))]

                    # and subselect events for this block
                    if blk < nr_blocks[-1]:
                        df_blk_ev = self.dataObj.events_df[(self.dataObj.events_df['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                                            (self.dataObj.events_df['onset'] >= blk_start_onsets[blk]) & \
                                                            (self.dataObj.events_df['onset'] < blk_start_onsets[blk+1])]
                    else:
                        df_blk_ev = self.dataObj.events_df[(self.dataObj.events_df['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                                            (self.dataObj.events_df['onset'] >= blk_start_onsets[blk])]
                        
                    # trial numbers for this blk
                    trials_blk = df_blk_info['index'].values

                    # loop over those trials
                    for t in trials_blk:

                        # events df for that trial
                        ev_df = df_blk_ev[df_blk_ev['trial_nr'] == t] 

                        ## if participant responded
                        if 'response' in ev_df['event_type'].values:

                            # participant response key
                            response_key = ev_df[ev_df['event_type'] == 'response']['response'].values[0]

                            # if correct response
                            if response_key[0] == df_blk_info[df_blk_info['index'] == t]['target_dot'].values[0][0].lower():
                                
                                correct_response = 1
                            
                            # simply incorrect
                            else:
                                correct_response = 0
                                
                            # save reaction time value
                            rt = ev_df[ev_df['event_type'] == 'response']['onset'].values[0] - ev_df[ev_df['event_type'] == 'stim']['onset'].values[0]

                            # if rt too low, exclude
                            if rt <= threshold_sec:
                                rt = np.nan
                                correct_response = np.nan

                            # save dataframe
                            df_manual_responses = pd.concat([df_manual_responses, 
                                                    pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)],  
                                                                'trial_num': [int(t)],
                                                                'block_num': [int(blk)], 
                                                                'target_ecc': [df_blk_info[df_blk_info['index'] == t]['target_ecc'].values[0]], 
                                                                'set_size': [df_blk_info[df_blk_info['index'] == t]['set_size'].values[0]],
                                                                'RT': [rt], 
                                                                'correct_response': [correct_response]})
                                                ])

                        else: # no responses given, missed trial
                            # save dataframe
                            df_manual_responses = pd.concat([df_manual_responses, 
                                                    pd.DataFrame({'sj':['sub-{sj}'.format(sj = pp)], 
                                                                'trial_num': [int(t)],
                                                                'block_num': [int(blk)], 
                                                                'target_ecc': [df_blk_info[df_blk_info['index'] == t]['target_ecc'].values[0]], 
                                                                'set_size': [df_blk_info[df_blk_info['index'] == t]['set_size'].values[0]],
                                                                'RT': [np.nan], 
                                                                'correct_response': [np.nan]})
                                                ])

                ## if too many missed trials, then should exclude participant
                pp_trials_resp = df_manual_responses[df_manual_responses.sj == 'sub-{sj}'.format(sj = pp)].correct_response.values
                pp_trials_missed = np.sum(np.isnan(pp_trials_resp))/len(pp_trials_resp)

                print('missed %.2f %% of trials'%(pp_trials_missed*100))

                if  pp_trials_missed >= missed_trl_thresh:
                    print('EXCLUDE')
                    self.exclude_sj_bool['sub-{sj}'.format(sj = pp)] = True

        ## if we want to exclude trial outliers (RT too long or fast)
        if exclude_outliers:
            self.df_manual_responses = self.exclude_search_outlier_trials(df_manual_responses, threshold_std = threshold_std)
        else:
            self.df_manual_responses = df_manual_responses

    def exclude_search_outlier_trials(self, df_manual_responses, threshold_std = 3):

        """helper function to
        exclude trials that have a RT of more than X standard deviation from the mean
        Note - assumes data has a normal distribution
        """

        clean_manual_responses = pd.DataFrame([])

        ## loop over subjects
        for pp in self.dataObj.sj_num:

            print('checking outlier reaction times for search task of sub-{sj}'.format(sj = pp))

            # loop over ecc
            for e in self.dataObj.ecc:
                
                for ss in self.dataObj.set_size:
                    
                    # sub select dataframe for specific set size and ecc
                    df_e_ss = df_manual_responses[(df_manual_responses['target_ecc'] == e) & \
                                            (df_manual_responses['set_size'] == ss) & \
                                            (df_manual_responses['correct_response'] == 1) & \
                                            (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]

                    # calculate mean RT for ecc and set size, and define outlier threshold value
                    all_rt = df_e_ss.RT.values

                    upper_thresh = np.nanmean(all_rt) + threshold_std * np.std(all_rt)
                    lower_thresh = np.nanmean(all_rt) - threshold_std * np.std(all_rt)

                    # first concatenate incorrect responses
                    clean_manual_responses = pd.concat((clean_manual_responses,  
                                                        df_manual_responses[(df_manual_responses['target_ecc'] == e) & \
                                                        (df_manual_responses['set_size'] == ss) & \
                                                        (df_manual_responses['correct_response'] != 1) & \
                                                        (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]), ignore_index=True)

                    # then concatenate dataframe rows that are not outliers
                    for index,row in df_e_ss.iterrows():
                        if (row.RT > upper_thresh) or (row.RT < lower_thresh):
                            row['RT'] = np.nan
                            row['correct_response'] = np.nan
                        clean_manual_responses = pd.concat((clean_manual_responses,  
                                                            pd.DataFrame(dict(row),index=[0])), ignore_index=True)
                    

        return clean_manual_responses
            
    def get_meanRT(self, df_manual_responses, acc_set_thresh = .75, acc_total_thresh = .85):
        
        """
        Given subject reponses and trial info, 
        returns data frame with mean reaction time and accuracy for each condition
        
        Parameters
        ----------
        df_manual_responses : DataFrame
            dataframe with results from get_RTs()
        
        """ 
        
        ####### if crowding task ###########
        if self.dataObj.task_name.lower() == 'crowding':
        
            # set up empty df
            df_mean_results = pd.DataFrame({'sj': [],'crowding_type': [],  'mean_RT': [], 
                       'accuracy': [], 'accuracy_color': [], 'accuracy_ori': []})
            
            # loop over subjects
            for pp in self.dataObj.sj_num:

                print('averaging reaction times for crowding task of sub-{sj}'.format(sj = pp))

                # loop over crowding types
                for crwd_type in self.crowding_type_all:
                
                    # correct trials
                    corrc_trls = df_manual_responses[(df_manual_responses['crowding_type'] == crwd_type) & \
                                                    (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['correct_response'].values

                    # accuracy
                    if crwd_type == 'unflankered':
                        accuracy = np.nansum(corrc_trls)/self.dataObj.nr_trials_unflank
                        accuracy_color = np.nansum(df_manual_responses[(df_manual_responses['crowding_type'] == crwd_type) & \
                                                    (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['correct_color'].values)/self.dataObj.nr_trials_unflank
                        accuracy_ori = np.nansum(df_manual_responses[(df_manual_responses['crowding_type'] == crwd_type) & \
                                                    (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['correct_ori'].values)/self.dataObj.nr_trials_unflank
                    else:
                        accuracy = np.nansum(corrc_trls)/self.dataObj.nr_trials_flank
                        accuracy_color = np.nansum(df_manual_responses[(df_manual_responses['crowding_type'] == crwd_type) & \
                                                    (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['correct_color'].values)/self.dataObj.nr_trials_flank
                        accuracy_ori = np.nansum(df_manual_responses[(df_manual_responses['crowding_type'] == crwd_type) & \
                                                    (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['correct_ori'].values)/self.dataObj.nr_trials_flank
                        
                    # mean RT
                    mean_RT = np.nanmean(df_manual_responses[(df_manual_responses['crowding_type'] == crwd_type) & \
                                                (df_manual_responses['correct_response'] == 1) & \
                                                (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['RT'].values)
                    
                    # save dataframe
                    df_mean_results = pd.concat([df_mean_results, 
                                            pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                        'crowding_type': [crwd_type],  
                                                        'mean_RT': [mean_RT], 
                                                        'accuracy': [accuracy], 
                                                        'accuracy_color': [accuracy_color], 
                                                        'accuracy_ori': [accuracy_ori]})
                                        ])
                
        ####### if search task ########
        elif self.dataObj.task_name.lower() == 'visualsearch':
            
            # set up empty df
            df_mean_results = pd.DataFrame({'sj': [],'target_ecc': [], 'set_size': [], 'mean_RT': [], 'accuracy': []})

            # loop over subjects
            for pp in self.dataObj.sj_num:

                print('averaging reaction times for search task of sub-{sj}'.format(sj = pp))

                # loop over ecc
                for e in self.dataObj.ecc:
                    
                    for ss in self.dataObj.set_size:
                        
                        # sub select dataframe for specific set size and ecc
                        df_e_ss = df_manual_responses[(df_manual_responses['target_ecc'] == e) & \
                                                (df_manual_responses['set_size'] == ss) & \
                                                (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]
                        
                        # accuracy
                        accuracy = np.nansum(df_e_ss['correct_response'].values)/len(df_e_ss['correct_response'].values)
                        
                        # mean reaction times
                        mean_RT = np.nanmean(df_e_ss[df_e_ss['correct_response'] == 1]['RT'].values)
                        
                        # save dataframe
                        df_mean_results = pd.concat([df_mean_results, 
                                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                            'target_ecc': [e], 
                                                            'set_size': [ss],
                                                            'mean_RT': [mean_RT], 
                                                            'accuracy': [accuracy]})
                                            ])

                ## if accuracy too low, then should exclude participant
                pp_acc_set = [np.mean(df_mean_results[(df_mean_results.sj == 'sub-{sj}'.format(sj = pp)) & \
                    (df_mean_results.set_size == ss)].accuracy.values) for ss in self.dataObj.set_size]
                
                if any(np.array(pp_acc_set) <= acc_set_thresh) or np.mean(pp_acc_set) <= acc_total_thresh:
                    print('search accuracy for each set size is %s %%, bellow thresh'%(str(np.array(pp_acc_set)*100)))
                    print('EXCLUDE')
                    self.exclude_sj_bool['sub-{sj}'.format(sj = pp)] = True
    
        self.df_mean_results = df_mean_results 
  
    def get_NoFlankers_meanRT(self, df_manual_responses, 
                                    feature_type = ['target_both', 'target_color', 'target_ori'],
                                    acc_thresh = .25):
        
        """
        Given subject reponses and trial info, 
        returns data frame with mean reaction time and accuracy for NO flankers trials
        
        Parameters
        ----------
        df_manual_responses : DataFrame
            dataframe with results from get_RTs()
        
        """   

        # set up empty df
        df_NoFlanker_results = pd.DataFrame({'sj': [],'type': [],  'mean_RT': [], 'accuracy': []})
        
        # loop over subjects
        for pp in self.dataObj.sj_num:

            print('averaging reaction times for UNcrowded trials of sub-{sj}'.format(sj = pp))

            # loop over crowding types
            for f_type in feature_type:
            
                # correct trials
                if f_type == 'target_color':
                    RT_f_type = df_manual_responses[(df_manual_responses['crowding_type'] == 'unflankered') & \
                                      (df_manual_responses['correct_color'] == 1) & \
                                      (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['RT'].values
                
                elif f_type == 'target_ori':
                    RT_f_type = df_manual_responses[(df_manual_responses['crowding_type'] == 'unflankered') & \
                                      (df_manual_responses['correct_ori'] == 1) & \
                                      (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['RT'].values

                elif f_type == 'target_both':
                    RT_f_type = df_manual_responses[(df_manual_responses['crowding_type'] == 'unflankered') & \
                                      (df_manual_responses['correct_response'] == 1) & \
                                      (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]['RT'].values

                
                accuracy = len(RT_f_type)/self.dataObj.nr_trials_unflank

                if accuracy <= acc_thresh and f_type == 'target_both':
                    print('accuracy for no flanker trials %.2f %%, thus below chance'%(accuracy*100))
                    print('EXCLUDE')
                    self.exclude_sj_bool['sub-{sj}'.format(sj = pp)] = True
                
                # save dataframe
                df_NoFlanker_results = pd.concat([df_NoFlanker_results, 
                                        pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                    'type': [f_type],  
                                                    'mean_RT': [np.nanmean(RT_f_type)], 
                                                    'accuracy': [accuracy]})
                                    ])
            
        self.df_NoFlanker_results = df_NoFlanker_results 
        
    def get_critical_spacing(self, staircases = None, num_trials = 96, cs_min_thresh = .20, cs_max_thresh = .65):
        
        """
        calculate critical spacing
        
        Parameters
        ----------
        staircases : DataFrame/dict
            dataframe with staircase intensities for all crowding types
        num_trials: int
            number of trials to use for CS calculation (last X trials)
        cs_min_thresh: float
            lower bound of CS, to use as exclusion criteria
        cs_max_thresh: float
            upper bound of CS, to use as exclusion criteria
        
        """ 
        
        
        # if not staircases provided 
        if staircases is None: 
            
            # load them from data object
            staircases = self.dataObj.load_staircases()

        self.staircases = staircases
        
        # set empty df
        df_CS = pd.DataFrame({'sj': [],'crowding_type': [],  'critical_spacing': []})

        # loop over subjects
        for pp in self.dataObj.sj_num:

            print('calculating critical spacing for sub-{sj}'.format(sj = pp))

            cs_val = []
            # loop over crowding types
            for crwd_type in self.dataObj.crwd_type:
                
                cs_val.append(np.median(self.staircases[(self.staircases['sj'] == 'sub-{sj}'.format(sj = pp))][crwd_type][-int(num_trials):]))

            ## check if cs median values lead to exclusion
            if np.median(cs_val) < cs_min_thresh or np.median(cs_val) > cs_max_thresh:
                print('median CS across crowding types is %.2f'%np.median(cs_val))
                print('EXCLUDE')
                self.exclude_sj_bool['sub-{sj}'.format(sj = pp)] = True
                
            # save df
            df_CS = pd.concat([df_CS, 
                                pd.DataFrame({'sj': np.tile('sub-{sj}'.format(sj = pp), len(self.dataObj.crwd_type)), 
                                            'crowding_type': self.dataObj.crwd_type,   
                                            'critical_spacing': cs_val})]) 

        self.df_CS = df_CS
   
    def combine_CS_df(self, df_CS):

        """
        Helper function to combine CS values
        in a same dataframe, in format that allows for across subject correlation

        """

        # build tidy dataframe with relevant info
        corr_df4plotting = pd.DataFrame([])

        # loop over subjects
        for _, pp in enumerate(df_CS.sj.unique()):
            
            tmp_df = df_CS[(df_CS['sj'] == pp)]
            
            corr_df4plotting = pd.concat((corr_df4plotting,
                                        pd.DataFrame({'sj': [pp],
                                                        'orientation': tmp_df[tmp_df['crowding_type'] == 'orientation'].critical_spacing.values,
                                                        'color': tmp_df[tmp_df['crowding_type'] == 'color'].critical_spacing.values,
                                                        'conjunction': tmp_df[tmp_df['crowding_type'] == 'conjunction'].critical_spacing.values
                                        })), ignore_index = True)
    
        return corr_df4plotting

    def get_search_slopes(self, df_manual_responses, per_ecc = True, fit_type = 'linear'):

        """
        calculate search slopes and intercept per ecc

        Parameters
        ----------
        df_manual_responses : DataFrame
            dataframe with results from get_RTs()
        per_ecc: bool
            if we want to get slope per eccentricity or combined over all
        """ 

        # set empty df
        df_search_slopes = pd.DataFrame({'sj': [],'target_ecc': [],  'slope': [], 'intercept': []})

        # loop over subjects
        for pp in self.dataObj.sj_num:

            print('calculating search slopes for sub-{sj}'.format(sj = pp))

            if per_ecc: # loop over ecc
                for e in self.dataObj.ecc: 
                    # sub-select df
                    df_temp = df_manual_responses[(df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                        (df_manual_responses['target_ecc'] == e) & \
                                        (df_manual_responses['correct_response'] == 1)]

                    if fit_type == 'linear':
                        # fit linear regressor
                        regressor = LinearRegression()
                        regressor.fit(df_temp[['set_size']], df_temp[['RT']]*1000) # because we want slope to be in ms/item
                        slope_val = regressor.coef_[0][0]
                        intercept = regressor.intercept_[0]

                    else: # assumes logarithmic
                        popt_, _ = scipy.optimize.curve_fit(lambda t,a,b: a+b*np.log(t), 
                                                                    df_temp.set_size.values, 
                                                                    df_temp.RT.values * 1000)
                        slope_val = popt_[1]
                        intercept = popt_[0]

                    # save df
                    df_search_slopes = pd.concat([df_search_slopes, 
                                                    pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                                'target_ecc': [e],   
                                                                'slope': [slope_val],
                                                                'intercept': [intercept]})])
            else:
                # sub-select df
                df_temp = df_manual_responses[(df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                    (df_manual_responses['correct_response'] == 1)]

                if fit_type == 'linear':
                    # fit linear regressor
                    regressor = LinearRegression()
                    regressor.fit(df_temp[['set_size']], df_temp[['RT']]*1000) # because we want slope to be in ms/item
                    slope_val = regressor.coef_[0][0]
                    intercept = regressor.intercept_[0]
                
                else: # assumes logarithmic
                        popt_, _ = scipy.optimize.curve_fit(lambda t,a,b: a+b*np.log(t), 
                                                                    df_temp.set_size.values, 
                                                                    df_temp.RT.values * 1000)
                        slope_val = popt_[1]
                        intercept = popt_[0]
                    
                # save df
                df_search_slopes = pd.concat([df_search_slopes, 
                                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                            'slope': [slope_val],
                                                            'intercept': [intercept]})])

        self.df_search_slopes = df_search_slopes 

    def split_half_search_df(self, df_manual_responses, groub_by = ['target_ecc', 'set_size'], seed_num = 1):

        """
        Quick function to split response dataframe into 2 random halfs
        To later calculate reliability
        """

        df_p1 = pd.DataFrame({})
        df_p2 = pd.DataFrame({})

        # loop over subjects
        for ind, pp in enumerate(self.dataObj.sj_num):

            print('spliting in 2 random halfs DF for sub-{sj}'.format(sj = pp))

            half_1 = df_manual_responses[(df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                        (df_manual_responses['correct_response'] == 1)].groupby(groub_by).sample(frac = 0.5,
                                                                                                                random_state = seed_num + ind)
            
            half_2 = df_manual_responses[(df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                        (df_manual_responses['correct_response'] == 1)].drop(half_1.index)
            
            # append
            df_p1 = pd.concat((df_p1, half_1), ignore_index=True)
            df_p2 = pd.concat((df_p2, half_2), ignore_index=True)

        return df_p1, df_p2

    def get_feature_acc_diff(self, df_mean_results = None):

        """
        calculate accuracy of crowding task by features,
        showing differences between unflakered and conditions, which will better account for inter-sub variability
        """

        # save accuracy diff for all participants in dataframe
        ACC_DIFF = pd.DataFrame({'sj': [], 'crowding_type': [], 'acc_diff_color': [], 'acc_diff_ori': []})

        # select unflankered condition
        unflanked_df = df_mean_results[df_mean_results['crowding_type'] == 'unflankered']

        for ct in ['color', 'conjunction', 'orientation']:
            
            # flanker condition
            flanked_df = df_mean_results[df_mean_results['crowding_type'] == ct]

            # accuracy difference for color
            acc_diff_color = ((flanked_df.accuracy_color.values / unflanked_df.accuracy_color.values) - 1) * 100
            # and for orientation
            acc_diff_ori = ((flanked_df.accuracy_ori.values / unflanked_df.accuracy_ori.values) - 1) * 100
            
            ACC_DIFF = pd.concat((ACC_DIFF,
                                pd.DataFrame({'sj': flanked_df.sj.values,
                                            'crowding_type': np.repeat(ct, len(flanked_df.sj.values)),
                                            'acc_diff_color': acc_diff_color, 
                                            'acc_diff_ori': acc_diff_ori
                                            })))

        return ACC_DIFF

    def calc_RT_split_half_reliability(self, df_manual_responses = None, iterations = 1000, seed_num = 29,
                                            return_slopes_arr = False):

        """
        Calculate split-half reliability of search RT slopes, over iterations 
        """

        ## split half a few times, and 
        # save values

        all_slopes_p1 = []
        all_slopes_p2 = []

        rho_all = []

        for i in range(iterations):

            ## split half RT dataframe

            half_1, half_2 = self.split_half_search_df(df_manual_responses, 
                                                groub_by = ['target_ecc', 'set_size'], 
                                                seed_num = seed_num+i)

            ## get slopes for each half
            self.get_search_slopes(df_manual_responses = half_1, per_ecc = False)
            df_search_slopes_p1 = self.df_search_slopes

            self.get_search_slopes(df_manual_responses = half_2, per_ecc = False)
            df_search_slopes_p2 = self.df_search_slopes
            
            
            ## now correlate both and see distribution
            rho, _ = scipy.stats.spearmanr(df_search_slopes_p1.slope.values, df_search_slopes_p2.slope.values)
            
            ## append all iteration values
            all_slopes_p1.append(df_search_slopes_p1.slope.values)
            all_slopes_p2.append(df_search_slopes_p2.slope.values)
            rho_all.append(rho)

        if return_slopes_arr:
            np.array(rho_all), df_search_slopes_p1, df_search_slopes_p2
        else:
            return np.array(rho_all)


    def calc_rsq(self, data_arr, pred_arr):
        return np.nan_to_num(1 - (np.nansum((data_arr - pred_arr)**2, axis=0)/ np.nansum(((data_arr - np.mean(data_arr))**2), axis=0)))

    def calc_chisq(self, data, pred, error = None):

        """
        Calculate model fit  chi-square
        
        Parameters
        ----------
        data : arr
            data array that was fitted
        pred : arr
            model prediction
        error: arr
            data uncertainty (if None will not be taken into account)
        """ 
    
        # residuals
        resid = data - pred 
        
        # if not providing uncertainty in ydata
        if error is None:
            error = np.ones(len(data))
        
        chisq = sum((resid/ error) ** 2)
        
        return chisq

    def calc_reduced_chisq(self, data, pred, error = None, n_params = None):

        """
        Calculate model fit Reduced chi-square
        
        Parameters
        ----------
        data : arr
            data array that was fitted
        pred : arr
            model prediction
        error: arr
            data uncertainty (if None will not be taken into account)
        n_params: int
            number of parameters in model
        """ 
        
        return self.calc_chisq(data, pred, error = error) / (len(data) - n_params)

    def calc_AIC(self, data, pred, error = None, n_params = None):

        """
        Calculate model fit Akaike Information Criterion,
        which measures of the relative quality for a fit, 
        trying to balance quality of fit with the number of variable parameters used in the fit
        
        Parameters
        ----------
        data : arr
            data array that was fitted
        pred : arr
            model prediction
        error: arr
            data uncertainty (if None will not be taken into account)
        n_params: int
            number of parameters in model
        """ 
        
        chisq = self.calc_chisq(data, pred, error = error)
        n_obs = len(data) # number of data points
        
        return n_obs * np.log(chisq/n_obs) + 2 * n_params
    
    def calc_BIC(self, data, pred, error = None, n_params = None):

        """
        Calculate model fit Bayesian information criterion,
        which measures of the relative quality for a fit, 
        trying to balance quality of fit with the number of variable parameters used in the fit
        
        Parameters
        ----------
        data : arr
            data array that was fitted
        pred : arr
            model prediction
        error: arr
            data uncertainty (if None will not be taken into account)
        n_params: int
            number of parameters in model
        """ 
        
        chisq = self.calc_chisq(data, pred, error = error)
        n_obs = len(data) # number of data points
        
        return n_obs * np.log(chisq/n_obs) + np.log(n_obs) * n_params
    
    def get_search_slope_prediction(self, df_coeff, data = None,
                                    fit_type = 'linear', dm = np.array([7,16,31])):
        
        """
        Quick function to get prediction array for search slopes
        given previously fitted model parameters (log vs lin) 
        """

        if fit_type == 'linear':
            pred_arr = df_coeff.slope.values[0] * dm + df_coeff.intercept.values[0]
        
        else: # assumes logarithmic
            pred_arr = df_coeff.slope.values[0] * np.log(dm) + df_coeff.intercept.values[0] 

        if data is not None:
            r2 = self.calc_rsq(data, pred_arr)
            return pred_arr, r2
        else:
            return pred_arr
    
    def make_search_CS_corr_2Dmatrix(self, corr_df4plotting, method = 'pearson'):

        """
        Helper function to make a correlation matrix,
        That summarizes info in dataframe that can be used for heatmap plotting

        Note - corr_df4plotting should have column names x_val and y_val for each set size and ecc,
        and those will be correlated. This allow for func to be used in several cases
        """

        corr_df = pd.DataFrame([])
        pval_df = pd.DataFrame([])

        for e_ind, ecc in enumerate(self.dataObj.ecc):
            
            tmp_corr_df = pd.DataFrame([])
            tmp_pval_df = pd.DataFrame([])
            
            for ss_ind, ss in enumerate(self.dataObj.set_size):

                if method == 'pearson':
                    rho, pval = scipy.stats.pearsonr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].y_val.values, 
                                        corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].x_val.values)
                elif method == 'spearman':
                    rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].y_val.values, 
                                        corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].x_val.values)
                
                tmp_corr_df = pd.concat((tmp_corr_df,
                                    pd.DataFrame([rho], 
                                                index=['{ss} items'.format(ss=ss)], 
                                                columns=['{ee} ecc'.format(ee=ecc)])
                                    ),axis=0)
                tmp_pval_df = pd.concat((tmp_pval_df,
                                    pd.DataFrame([pval], 
                                                index=['{ss} items'.format(ss=ss)], 
                                                columns=['{ee} ecc'.format(ee=ecc)])
                                    ),axis=0)
            
            corr_df =  pd.concat((corr_df,tmp_corr_df),axis=1)
            pval_df =  pd.concat((pval_df,tmp_pval_df),axis=1)

        return corr_df, pval_df

    def make_search_CS_corr_1Dmatrix(self, corr_df4plotting, method = 'pearson'):

        """
        Helper function to make a correlation matrix,
        That summarizes info in dataframe that can be used for heatmap plotting

        Note - corr_df4plotting should have column names x_val and y_val for each set size and ecc,
        and those will be correlated. This allow for func to be used in several cases
        """

        corr_df = pd.DataFrame([])
        pval_df = pd.DataFrame([])

        for e_ind, ecc in enumerate(self.dataObj.ecc):

            if method == 'pearson':
                rho, pval = scipy.stats.pearsonr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc)].y_val.values, 
                                    corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc)].x_val.values)
            elif method == 'spearman':
                rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc)].y_val.values, 
                                    corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc)].x_val.values)
        
            corr_df = pd.concat((corr_df,
                             pd.DataFrame([rho], 
                                          columns=['{ee} ecc'.format(ee=ecc)])
                            ),axis=1)
            pval_df = pd.concat((pval_df,
                                    pd.DataFrame([pval], 
                                                columns=['{ee} ecc'.format(ee=ecc)])
                                    ),axis=1)

        return corr_df, pval_df


    def get_OriVSdata_meanRT(self, ecc = [4, 8, 12], setsize = [5,15,30], minRT = .250, max_RT = 5):

        """
        Helper function to calculate mean RT for search data
        of previous experiment. Based on what was done in other repo
        """

        ## Load summary measures
        previous_dataset_pth = self.dataObj.params['paths']['data_ori_vs_pth']
        sum_measures = np.load(op.join(previous_dataset_pth,
                                    'summary','sum_measures.npz')) # all relevant measures

        ## get list with all participants
        # that passed the inclusion criteria
        previous_sub_list = [val for val in sum_measures['all_subs'] if val not in sum_measures['excluded_sub']]

        # set up empty df
        df_mean_results = pd.DataFrame({'sj': [],'target_ecc': [], 'set_size': [], 'mean_RT': []})

        # loop over participants
        for prev_pp in previous_sub_list:
            # load behav data search
            prev_pp_VSfile = op.join(previous_dataset_pth, 'output_VS', 
                                        'data_visualsearch_pp_{p}.csv'.format(p = prev_pp))
            df_vs = pd.read_csv(prev_pp_VSfile, sep='\t')

            # loop over ecc and set size
            for e in ecc:
                for ss in setsize:

                    # sub select dataframe for specific set size and ecc
                    df_e_ss = df_vs[(df_vs['target_ecc'] == e) & \
                                    (df_vs['set_size'] == ss) & \
                                    (df_vs['key_pressed'] == df_vs['target_orientation']) & \
                                    (df_vs['RT'] > minRT) & \
                                    (df_vs['RT'] < max_RT)]


                    df_mean_results = pd.concat((df_mean_results,
                                                pd.DataFrame({'sj': [prev_pp],
                                                            'target_ecc': [e], 
                                                            'set_size': [ss], 
                                                            'mean_RT': [np.nanmean(df_e_ss.RT.values)]})), 
                                                            ignore_index = True)

        return df_mean_results

    def get_OriVSdata_CS(self, ecc = [4, 8, 12]):
        
        """
        Helper function to CS values for crowding data
        of previous experiment. Based on what was done in other repo
        """

        ## Load summary measures
        previous_dataset_pth = self.dataObj.params['paths']['data_ori_vs_pth']
        sum_measures = np.load(op.join(previous_dataset_pth,
                                    'summary','sum_measures.npz')) # all relevant measures

        ## get list with all participants
        # that passed the inclusion criteria
        previous_sub_list = [val for val in sum_measures['all_subs'] if val not in sum_measures['excluded_sub']]

        ## get CS values for all eccentricities 
        cs_val_list = [sum_measures['all_cs'][ind] for ind, val in enumerate(sum_measures['all_subs']) if val not in sum_measures['excluded_sub']]

        ## save in dataframe
        df_prev_CS = pd.DataFrame({'sj': np.repeat(previous_sub_list, len(ecc)),
                                  'CS': np.array(cs_val_list).ravel(),
                                  'CS_ecc': np.tile(ecc, len(previous_sub_list))})

        return df_prev_CS

    def get_OriVSdata_slopes(self, minRT = .250, max_RT = 5):

        """
        Helper function to calculate RT slopes for search data
        of previous experiment. Based on what was done in other repo
        """

        ## Load summary measures
        previous_dataset_pth = self.dataObj.params['paths']['data_ori_vs_pth']
        sum_measures = np.load(op.join(previous_dataset_pth,
                                    'summary','sum_measures.npz')) # all relevant measures

        ## get list with all participants
        # that passed the inclusion criteria
        previous_sub_list = [val for val in sum_measures['all_subs'] if val not in sum_measures['excluded_sub']]

        # set up empty df
        df_slope_results = pd.DataFrame({'sj': [], 'slope': [], 'intercept': []})

        # loop over participants
        for prev_pp in previous_sub_list:
            # load behav data search
            prev_pp_VSfile = op.join(previous_dataset_pth, 'output_VS', 
                                        'data_visualsearch_pp_{p}.csv'.format(p = prev_pp))
            df_vs = pd.read_csv(prev_pp_VSfile, sep='\t')

            # sub select dataframe 
            df_tmp = df_vs[(df_vs['key_pressed'] == df_vs['target_orientation']) & \
                        (df_vs['RT'] > minRT) & \
                        (df_vs['RT'] < max_RT)]

            # fit linear regressor
            regressor = LinearRegression()
            regressor.fit(df_tmp[['set_size']], df_tmp[['RT']]*1000) # because we want slope to be in ms/item

            # save df
            df_slope_results = pd.concat((df_slope_results, 
                                        pd.DataFrame({'sj': [prev_pp],
                                                    'slope': [regressor.coef_[0][0]],
                                                    'intercept': [regressor.intercept_[0]]})), ignore_index = True)

        return df_slope_results