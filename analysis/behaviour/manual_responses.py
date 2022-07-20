import numpy as np
import os, sys
import os.path as op
import pandas as pd
import seaborn as sns
import yaml

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
        
    
    def get_RTs(self):
        
        """
        Given subject reponses and trial info, 
        returns data frame with reaction time for all trials
        
        """ 
        
        ####### if crowding task ###########
        if self.dataObj.task_name.lower() == 'crowding':
            
            # all crowding types (includes uncrowding)
            self.crowding_type_all = self.dataObj.trial_info_df['crowding_type'].unique()
            
            # set up empty df
            df_manual_responses = pd.DataFrame({'sj': [], 'trial_num': [] ,'crowding_type': [],  'RT': [], 
                       'correct_response': [], 'correct_color': [], 'correct_ori': []})
            
            # loop over crowding types
            for crwd_type in self.crowding_type_all:
                
                # subselect trial info df to only have trials from this crowding type
                df_crwd_type_info = self.dataObj.trial_info_df[self.dataObj.trial_info_df['crowding_type'] == crwd_type]
                
                # trial numbers for this crowding type
                trials_crwd_type = df_crwd_type_info['index'].values
                
                # loop over those trials
                for t in trials_crwd_type:
                    
                    # events df for that trial
                    ev_df = self.dataObj.events_df[self.dataObj.events_df['trial_nr'] == t] 

                    # if participant responded
                    if 'response' in ev_df['event_type'].values:

                        # participant response key
                        response_key = ev_df[ev_df['event_type'] == 'response']['response'].values[0]

                        # trial target name
                        trl_targt = df_crwd_type_info[df_crwd_type_info['index'] == t]['target_name'].values[0]

                        # if correct response
                        if response_key in self.dataObj.task_keys[trl_targt]:
                            correct_response = 1
                            correct_color = 1 
                            correct_ori = 1
                        # if correct color but not orientation
                        elif response_key in [self.dataObj.task_keys[l][0] for l in self.dataObj.task_keys if trl_targt[0] in l]:
                            correct_response = 0
                            correct_color = 1 
                            correct_ori = 0
                        # if correct orientation but not color
                        elif response_key in [self.dataObj.task_keys[l][0] for l in self.dataObj.task_keys if trl_targt[-1] in l]:
                            correct_response = 0
                            correct_color = 0
                            correct_ori = 1
                        # simply incorrect
                        else:
                            correct_response = 0
                            correct_color = 0
                            correct_ori = 0

                        # save reaction time value
                        rt = ev_df[ev_df['event_type'] == 'response']['onset'].values[0] - ev_df[ev_df['event_type'] == 'stim']['onset'].values[0]

                        # save dataframe
                        df_manual_responses = pd.concat([df_manual_responses, 
                                                pd.DataFrame({'sj': ['sub-'+self.dataObj.sj_num], 
                                                              'trial_num': [int(t)],
                                                              'crowding_type': [crwd_type],  
                                                              'RT': [rt], 
                                                               'correct_response': [correct_response], 
                                                              'correct_color': [correct_color], 
                                                              'correct_ori': [correct_ori]})
                                               ])

                    else: # no responses given, missed trial
                        df_manual_responses = pd.concat([df_manual_responses, 
                                                pd.DataFrame({'sj': ['sub-'+self.dataObj.sj_num], 
                                                              'trial_num': [int(t)],
                                                              'crowding_type': [crwd_type],  
                                                              'RT': [np.nan], 
                                                               'correct_response': [np.nan], 
                                                              'correct_color': [np.nan], 
                                                              'correct_ori': [np.nan]})
                                               ])

        ####### if search task ########
        elif self.dataObj.task_name.lower() == 'visualsearch':
            print('not implemented yet')
    
        
        self.df_manual_responses = df_manual_responses
        
        
        
    def get_meanRT(self, df_manual_responses):
        
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
            
            # loop over crowding types
            for crwd_type in self.crowding_type_all:
            
                # correct trials
                corrc_trls = df_manual_responses[df_manual_responses['crowding_type'] == crwd_type]['correct_response'].values

                # accuracy
                if crwd_type == 'unflankered':
                    accuracy = np.nansum(corrc_trls)/self.dataObj.nr_trials_unflank
                    accuracy_color = np.nansum(df_manual_responses[df_manual_responses['crowding_type'] == crwd_type]['correct_color'].values)/self.dataObj.nr_trials_unflank
                    accuracy_ori = np.nansum(df_manual_responses[df_manual_responses['crowding_type'] == crwd_type]['correct_ori'].values)/self.dataObj.nr_trials_unflank
                else:
                    accuracy = np.nansum(corrc_trls)/self.dataObj.nr_trials_flank
                    accuracy_color = np.nansum(df_manual_responses[df_manual_responses['crowding_type'] == crwd_type]['correct_color'].values)/self.dataObj.nr_trials_flank
                    accuracy_ori = np.nansum(df_manual_responses[df_manual_responses['crowding_type'] == crwd_type]['correct_ori'].values)/self.dataObj.nr_trials_flank
                    
                # mean RT
                mean_RT = np.nanmean(df_manual_responses[(df_manual_responses['crowding_type'] == crwd_type)&\
                                             (df_manual_responses['correct_response'] == 1)]['RT'].values)
                
                # save dataframe
                df_mean_results = pd.concat([df_mean_results, 
                                        pd.DataFrame({'sj': ['sub-'+self.dataObj.sj_num], 
                                                      'crowding_type': [crwd_type],  
                                                      'mean_RT': [mean_RT], 
                                                      'accuracy': [accuracy], 
                                                      'accuracy_color': [accuracy_color], 
                                                      'accuracy_ori': [accuracy_ori]})
                                       ])
                
        ####### if search task ########
        elif self.dataObj.task_name.lower() == 'visualsearch':
            print('not implemented yet')
    
    
        
        self.df_mean_results = df_mean_results   
        

    def get_critical_spacing(self, staircases = None, num_trials = 96):
        
        """
        calculate critical spacing
        
        Parameters
        ----------
        staircases : DataFrame/dict
            dataframe with staircase intensities for all crowding types
        
        """ 
        
        
        # if not staircases provided 
        if staircases is None: 
            
            # load them from data object
            staircases = self.dataObj.load_staircases()
        
        cs_val = []
        # loop over crowding types
        for crwd_type in self.dataObj.crwd_type:
            
            cs_val.append(np.mean(staircases[crwd_type][-int(num_trials):]))
            
        # save df
        self.df_CS = pd.DataFrame({'sj': np.tile('sub-'+self.dataObj.sj_num, len(self.dataObj.crwd_type)), 
                                    'crowding_type': self.dataObj.crwd_type,   
                                   'critical_spacing': cs_val})
            
