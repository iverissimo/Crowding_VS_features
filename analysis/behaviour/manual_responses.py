import numpy as np
import os, sys
import os.path as op
import pandas as pd
import seaborn as sns
import yaml

from sklearn.linear_model import LinearRegression

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

    
    def get_RTs(self, missed_trl_thresh = .25):
        
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

        self.df_manual_responses = df_manual_responses
        
        
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


    def get_search_slopes(self, df_manual_responses):

        """
        calculate search slopes and intercept per ecc

        Parameters
        ----------
        df_manual_responses : DataFrame
            dataframe with results from get_RTs()
        
        """ 

        # set empty df
        df_search_slopes = pd.DataFrame({'sj': [],'target_ecc': [],  'slope': [], 'intercept': []})

        # loop over subjects
        for pp in self.dataObj.sj_num:

            print('calculating search slopes for sub-{sj}'.format(sj = pp))

            # loop over ecc
            for e in self.dataObj.ecc: 

                # sub-select df
                df_temp = df_manual_responses[(df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                    (df_manual_responses['target_ecc'] == e) & \
                                    (df_manual_responses['correct_response'] == 1)]

                # fit linear regressor
                regressor = LinearRegression()
                regressor.fit(df_temp[['set_size']], df_temp[['RT']]*1000) # because we want slope to be in ms/item

                # save df
                df_search_slopes = pd.concat([df_search_slopes, 
                                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                            'target_ecc': [e],   
                                                            'slope': [regressor.coef_[0][0]],
                                                            'intercept': [regressor.intercept_[0]]})])

        self.df_search_slopes = df_search_slopes 

