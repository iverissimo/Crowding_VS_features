from string import whitespace
import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml

from ast import literal_eval

class BehTask:
    
    """Behavioral Task 
    Class that takes care of loading data paths and setting experiment params
    """
    
    def __init__(self, params, sj_num, session = 'test', exclude_sj = []):
        
        """__init__
        constructor for class, takes experiment params and subject num as input
        
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live, if not string assumes dict
        sj_num : str
            participant number
        session : str
            session type (test/train)
        exclude_sj: list/str
            list of subjects to exclude (can be none)
            
        """
        
        ## set params
        if isinstance(params, str):
            # load settings from yaml
            with open(params, 'r') as f_in:
                self.params = yaml.safe_load(f_in)
        else:
            self.params = params

        ## set some paths
        data_folder = self.params['paths']['data_pth'][self.params['paths']['curr_dir']]
        self.sourcedata_pth = op.join(data_folder, 'sourcedata')
        self.derivatives_pth = op.join(data_folder, 'derivatives')

        ## list of excluded subs
        self.excl_file = op.join(self.derivatives_pth, 'excluded_participants.csv')

        if op.isfile(self.excl_file):
            print('loading file with excluded participants ID')
            exclude_sj = pd.read_csv(self.excl_file, header = None)[0].values

        ## if there's participants we want to exclude
        if len(exclude_sj) > 0:
            self.exclude_sj = [str(s).zfill(3) for s in exclude_sj]
        else:
            self.exclude_sj = exclude_sj
            
        ## set sj number list
        # if we want all participants in sourcedata folder
        if sj_num in ['group', 'all']: 
            self.sj_num = [str(s[4:]).zfill(3) for s in os.listdir(self.sourcedata_pth) if 'sub-' in s and str(s[4:]).zfill(3) not in self.exclude_sj]
        
        # if we provide list of sj numbers
        elif isinstance(sj_num, list) or isinstance(sj_num, np.ndarray): 
            self.sj_num = [str(s).zfill(3) for s in sj_num if str(s).zfill(3) not in self.exclude_sj]
        
        # if only one participant, put in list to make life easier later
        else:
            self.sj_num = [str(sj_num).zfill(3)] 

        print('Total number of subjects %i'%len(self.sj_num))
        
        ##  and session type
        self.session = session
    

    def get_task_info(self, task_name):
        
        """
        Get task files (trial info csv) and return dataframe

        Parameters
        ----------
        task_name : str
            name of task we want to retrieve ('Crowding'/'VisualSearch')
        """

        for i, pp in enumerate(self.sj_num):

            ## subject data folder
            sj_pth = op.join(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj = pp)))
            
            ## load trial info, with step up for that session
            trial_info_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                        ses = self.session)) and \
                    'task-{task}'.format(task = task_name) in x and x.endswith('_trial_info.csv')]
            
            ## load dataframe
            if len(trial_info_files) != 1:
                raise ValueError('{num} info files found for sub-{sj} in task {tsk}!!'.format(num = len(trial_info_files),
                                                                                        sj = pp,
                                                                                        tsk = task_name))
            else:
                if i == 0:
                    trial_info_df = pd.read_csv(trial_info_files[0])
                    trial_info_df['sj'] = 'sub-{sj}'.format(sj = pp) # add participant identifier
                else: 
                    tmp_df = pd.read_csv(trial_info_files[0])
                    tmp_df['sj'] = 'sub-{sj}'.format(sj = pp) # add participant identifier

                    trial_info_df = pd.concat([trial_info_df, tmp_df])
                    
        return trial_info_df
        
    
    def get_task_events(self, task_name):
        
        """
        Get events files (events tsv) and return dataframe

        Parameters
        ----------
        task_name : str
            name of task we want to retrieve ('Crowding'/'VisualSearch')
        """

        for i, pp in enumerate(self.sj_num):

            ## subject data folder
            sj_pth = op.join(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj = pp)))
        
            ## load trial info, with step up for that session
            events_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                    ses = self.session)) and \
                            'task-{task}'.format(task = task_name) in x and x.endswith('_events.tsv')]
            
            ## load dataframe
            if len(events_files) != 1:
                raise ValueError('{num} event files found for sub-{sj} in task {tsk}!!'.format(num = len(events_files),
                                                                                        sj = pp,
                                                                                        tsk = task_name))
            else:
                if i == 0:
                    events_df = pd.read_csv(events_files[0], sep="\t")
                    events_df['sj'] = 'sub-{sj}'.format(sj = pp) # add participant identifier
                else: 
                    tmp_df = pd.read_csv(events_files[0], sep="\t")
                    tmp_df['sj'] = 'sub-{sj}'.format(sj = pp) # add participant identifier

                    events_df = pd.concat([events_df, tmp_df])

        return events_df

    
    def convert_df_column_from_str(self, df_column, whitespaces=True):

        """
        Helper function to convert dataframe column with strings of lists, to actual lists

        Parameters
        ----------
        df_column : pd dataframe column
        """
        
        # if there are whitespaces we want to remove 
        if whitespaces:
            new_column = df_column.str.replace('\s+]', ']', 
                            regex=True).str.replace('\[\s+', '[', 
                            regex=True).str.replace('\s+', ', ', 
                            regex=True).apply(literal_eval)
        else:
            new_column = df_column.apply(literal_eval)
        
        return new_column
        

class BehCrowding(BehTask):
   
    def __init__(self, params, sj_num, session = 'test', exclude_sj = []):  # initialize child class

        """ Initializes BehCrowding object. 
      
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str
            participant number
        session : str
            session type (test/train)
        exclude_sj: list/str
            list of subjects to exclude (can be none)
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, session = session, exclude_sj = exclude_sj)
        
        self.task_name = 'Crowding'
        
        ## some relevant params
        self.distance_bounds = self.params['crowding']['staircase']['distance_ratio_bounds'] # max and min distance ratio (ratio x ecc)
        self.crwd_type = self.params['crowding']['crwd_type'] # types of crowding
        self.ratio_trls_cs = self.params['crowding']['cs_trial_ratio'] # ratio of flanker trials to use for CS calculation

        ## task keys
        self.task_keys = self.params['keys']['target_key']
                         
        ## get events
        self.events_df = self.get_task_events(task_name = self.task_name)

        ## get trial info
        self.trial_info_df = self.get_task_info(task_name = self.task_name)   

        ## convert columns as needed to fix pd_to_csv list-str bug
        # (ugly but works - should improve later)
        for col_name in self.trial_info_df.columns:

            if isinstance(self.trial_info_df[col_name].values[0], str) and \
                len(self.trial_info_df[col_name].values[0]) > 1 and \
                    self.trial_info_df[col_name].values[0][:1] == '[':

                    print('updating %s'%col_name)

                    # some columns will need to correct whitespaces before literal eval
                    whitespace_bool = False if col_name in ['target_pos', 'target_color', \
                        'distractor_name', 'distractor_pos', 'distractor_color', 'distractor_ori'] else True
                    
                    self.trial_info_df[col_name] =  self.convert_df_column_from_str(self.trial_info_df[col_name], 
                                                            whitespaces = whitespace_bool)

                         
        ## number of trials
        self.nr_trials_flank = len(self.trial_info_df.loc[(self.trial_info_df['crowding_type'] == 'orientation') & \
                                                            (self.trial_info_df['sj'] == 'sub-{sj}'.format(sj = self.sj_num[0]))]['index'].values)
        self.nr_trials_unflank = len(self.trial_info_df.loc[(self.trial_info_df['crowding_type'] == 'unflankered') & \
                                                            (self.trial_info_df['sj'] == 'sub-{sj}'.format(sj = self.sj_num[0]))]['index'].values)
        print('Number of trials with flankers is {num_fl} and without is {num_unfl}'.format(num_fl = self.nr_trials_flank,
                                                                                            num_unfl = self.nr_trials_unflank))
    
    def load_staircases(self):
                         
        """
        Get staircase object used in task, 
        return dataframe with each staircase intensity values per participant in list

        """              
        
        for i, pp in enumerate(self.sj_num):

            ## subject data folder
            sj_pth = op.join(op.join(self.sourcedata_pth, 'sub-{sj}'.format(sj = pp)))
                         
            ## absolute path for staircase files
            staircase_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                    ses = self.session)) and \
                            'task-{task}'.format(task = self.task_name) in x and 'staircase' in x and x.endswith('.pickle')]
            
            ## load each staircase
            staircases = {}
            for crwd in self.crwd_type:
                staircases[crwd] = pd.read_pickle([val for val in staircase_files if crwd in val][0]).intensities
            
            ## save as dataframe for ease of use
            staircases = pd.DataFrame(staircases)
            staircases['sj'] = 'sub-{sj}'.format(sj = pp) # add participant identifier 

            if i == 0:    
                self.staircases = staircases
            else:
                self.staircases = pd.concat([self.staircases, staircases])

        return self.staircases
                         
                         
class BehVsearch(BehTask):
   
    def __init__(self, params, sj_num, session = 'test', exclude_sj = []):  # initialize child class

        """ Initializes BehVsearch object. 
      
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str
            participant number
        session : str
            session type (test/train)
        exclude_sj: list/str
            list of subjects to exclude (can be none)
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, session = session, exclude_sj = exclude_sj)
        
        self.task_name = 'VisualSearch'
        
        ## task keys
        self.task_keys = ['right', 'left']

        ## some relevant params
        self.ecc = self.params['visual_search']['num_ecc']
        self.set_size = self.params['visual_search']['set_size']
        self.target_names = self.params['visual_search']['target_names']
        
        ## get events
        self.events_df = self.get_task_events(task_name = self.task_name)
        
        ## get trial info
        self.trial_info_df = self.get_task_info(task_name = self.task_name)  

        ## convert columns as needed to fix pd_to_csv list-str bug
        # (ugly but works - should improve later)
        for col_name in self.trial_info_df.columns:

            if isinstance(self.trial_info_df[col_name].values[0], str) and \
                len(self.trial_info_df[col_name].values[0]) > 1 and \
                    self.trial_info_df[col_name].values[0][:1] == '[':

                    print('updating %s'%col_name)

                    # some columns will need to correct whitespaces before literal eval
                    whitespace_bool = False if col_name in ['target_color', 'target_dot', \
                        'distractor_color', 'distractor_ori', 'distractor_dot'] else True
                    
                    self.trial_info_df[col_name] =  self.convert_df_column_from_str(self.trial_info_df[col_name], 
                                                            whitespaces = whitespace_bool)




        