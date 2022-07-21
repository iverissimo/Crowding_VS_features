import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml


class BehTask:
    
    """Behavioral Task 
    Class that takes care of loading data paths and setting experiment params
    """
    
    def __init__(self, params, sj_num, session, exclude_sj = []):
        
        """__init__
        constructor for class, takes experiment params and subject num as input
        
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str
            participant number
        session : str
            session type (test/train)
            
        """
        
        # set params
        
        if isinstance(params, str):
            # load settings from yaml
            with open(params, 'r') as f_in:
                self.params = yaml.safe_load(f_in)
        else:
            self.params = params

        # set some paths
        self.data_path = self.params['paths']['data_pth'][self.params['paths']['curr_dir']]
        self.derivatives_pth = op.join(self.data_path,'derivatives')
            
        # set sj number
        if sj_num in ['group', 'all']: # if we want all participants in sourcedata folder
            self.sj_num = [str(s[4:]).zfill(3) for s in os.listdir(op.join(self.data_path, 'sourcedata')) if 'sub-' in s and str(s[4:]).zfill(3)  not in exclude_sj]
        
        elif isinstance(sj_num, list) or isinstance(sj_num, np.ndarray): # if we provide list of sj numbers
            self.sj_num = [str(s).zfill(3) for s in sj_num if str(s).zfill(3)  not in exclude_sj]
        
        else:
            self.sj_num = [str(sj_num).zfill(3)] # if only one participant, put in list to make life easier later
        
        #  and session type
        self.session = session
        self.exclude_sj = exclude_sj
    
    def get_task_info(self, task_name):
        
        """
        Get task files (trial info csv) and return dataframe
        """

        for i, pp in enumerate(self.sj_num):

            # subject data folder
            sj_pth = op.join(op.join(self.data_path, 'sourcedata', 'sub-{sj}'.format(sj = pp)))
            
            # load trial info, with step up for that session
            trial_info_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                        ses = self.session)) and \
                    'task-{task}'.format(task = task_name) in x and x.endswith('_trial_info.csv')]
            
            # load dataframe
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
        """

        for i, pp in enumerate(self.sj_num):

            # subject data folder
            sj_pth = op.join(op.join(self.data_path, 'sourcedata', 'sub-{sj}'.format(sj = pp)))
        
            # load trial info, with step up for that session
            events_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                    ses = self.session)) and \
                            'task-{task}'.format(task = task_name) in x and x.endswith('_events.tsv')]
            
            # load dataframe
            if i == 0:
                events_df = pd.read_csv(events_files[0], sep="\t")
                events_df['sj'] = 'sub-{sj}'.format(sj = pp) # add participant identifier
            else: 
                tmp_df = pd.read_csv(events_files[0], sep="\t")
                tmp_df['sj'] = 'sub-{sj}'.format(sj = pp) # add participant identifier

                events_df = pd.concat([events_df, tmp_df])

                         
        return events_df
        

class BehCrowding(BehTask):
   
    def __init__(self, params, sj_num, session, exclude_sj = []):  # initialize child class

        """ Initializes BehCrowding object. 
      
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str
            participant number
        session : str
            session type (test/train)
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, session = session, exclude_sj = exclude_sj)
        
        self.task_name = 'Crowding'
        
        # some relevant params
        self.distance_bounds = self.params['crowding']['staircase']['distance_ratio_bounds']
        self.crwd_type = self.params['crowding']['crwd_type'] # types of crowding
        self.ratio_trls_cs = 1/2 # ration of trials to use for CS calculation

        # task keys
        self.task_keys = self.params['keys']['target_key']
                         
        # get events
        self.events_df = self.get_task_events(task_name = self.task_name)
        # get trial info
        self.trial_info_df = self.get_task_info(task_name = self.task_name)   
                         
        # number of trials
        self.nr_trials_flank = len(self.trial_info_df.loc[(self.trial_info_df['crowding_type'] == 'orientation') & \
                                                            (self.trial_info_df['sj'] == 'sub-{sj}'.format(sj = self.sj_num[0]))]['index'].values)
        self.nr_trials_unflank = len(self.trial_info_df.loc[(self.trial_info_df['crowding_type'] == 'unflankered') & \
                                                            (self.trial_info_df['sj'] == 'sub-{sj}'.format(sj = self.sj_num[0]))]['index'].values)
        
    
    def load_staircases(self):
                         
        """
        Get staircase object used in task, return dict with each staircase intensity values
        """              
        
        for i, pp in enumerate(self.sj_num):

            # subject data folder
            sj_pth = op.join(op.join(self.data_path, 'sourcedata', 'sub-{sj}'.format(sj = pp)))
                         
            # absolute path for staircase files
            staircase_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                    ses = self.session)) and \
                            'task-{task}'.format(task = self.task_name) in x and 'staircase' in x and x.endswith('.pickle')]
            
            # load each staircase
            staircases = {}
            for crwd in self.crwd_type:
                staircases[crwd] = pd.read_pickle([val for val in staircase_files if crwd in val][0]).intensities
            
            # save as dataframe for ease of use
            staircases = pd.DataFrame(staircases)
            staircases['sj'] = 'sub-{sj}'.format(sj = pp) # add participant identifier 

            if i == 0:    
                self.staircases = staircases
            else:
                self.staircases = pd.concat([self.staircases, staircases])

        return self.staircases
                         
                         
class BehVsearch(BehTask):
   
    def __init__(self, params, sj_num, session, exclude_sj = []):  # initialize child class

        """ Initializes BehVsearch object. 
      
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str
            participant number
        session : str
            session type (test/train)
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, session = session, exclude_sj = exclude_sj)
        
        self.task_name = 'VisualSearch'
        
        # task keys
        self.task_keys = ['right', 'left']

        # some relevant params
        self.ecc = self.params['visual_search']['num_ecc']
        self.set_size = self.params['visual_search']['set_size']
        self.target_names = self.params['visual_search']['target_names']
        
        # get events
        self.events_df = self.get_task_events(task_name = self.task_name)
        # get trial info
        self.trial_info_df = self.get_task_info(task_name = self.task_name)   


        