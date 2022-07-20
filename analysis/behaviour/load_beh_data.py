import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml


class BehTask:
    
    """Behavioral Task 
    Class that takes care of loading data paths and setting experiment params
    """
    
    def __init__(self, params, sj_num, session):
        
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
            
        # set sj number
        self.sj_num = str(sj_num).zfill(3)
        self.session = session
        
        # set some paths
        self.data_path = self.params['paths']['data_pth'][self.params['paths']['curr_dir']]
        self.derivatives_pth = op.join(self.data_path,'derivatives')
        
    
    def get_task_info(self, task_name):
        
        """
        Get task files (trial info csv) and return dataframe
        """

        # subject data folder
        sj_pth = op.join(op.join(self.data_path, 'sourcedata', 'sub-{sj}'.format(sj = self.sj_num)))
        
        # load trial info, with step up for that session
        trial_info_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = self.sj_num, 
                                                                                                                    ses = self.session)) and \
                'task-{task}'.format(task = task_name) in x and x.endswith('_trial_info.csv')]
        
        # load dataframe
        trial_info_df = pd.read_csv(trial_info_files[0])
                         
        return trial_info_df
        
    
    def get_task_events(self, task_name):
        
        """
        Get events files (events tsv) and return dataframe
        """

        # subject data folder
        sj_pth = op.join(op.join(self.data_path, 'sourcedata', 'sub-{sj}'.format(sj = self.sj_num)))
        
        # load trial info, with step up for that session
        events_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = self.sj_num, ses = self.session)) and \
                        'task-{task}'.format(task = task_name) in x and x.endswith('_events.tsv')]
        # load dataframe
        events_df = pd.read_csv(events_files[0], sep="\t")
        # only select onsets > 0 (rest is intruction time)
        events_df = events_df[events_df['onset']>0]
                         
        return events_df
        

class BehCrowding(BehTask):
   
    def __init__(self, params, sj_num, session):  # initialize child class

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
        super().__init__(params = params, sj_num = sj_num, session = session)
        
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
        self.nr_trials_flank = len(self.trial_info_df.loc[self.trial_info_df['crowding_type']=='orientation']['index'].values)
        self.nr_trials_unflank = len(self.trial_info_df.loc[self.trial_info_df['crowding_type']=='unflankered']['index'].values)
        
    
    def load_staircases(self):
                         
        """
        Get staircase object used in task, return dict with each staircase intensity values
        """              
        # subject data folder
        sj_pth = op.join(op.join(self.data_path, 'sourcedata', 'sub-{sj}'.format(sj = self.sj_num)))
                         
        # absolute path for staircase files
        staircase_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = self.sj_num, 
                                                                                                                ses = self.session)) and \
                        'task-{task}'.format(task = self.task_name) in x and 'staircase' in x and x.endswith('.pickle')]
        # load each staircase, save in dict for ease
        staircases = {}
        for crwd in self.crwd_type:
            staircases[crwd] = pd.read_pickle([val for val in staircase_files if crwd in val][0]).intensities
                         
        self.staircases = staircases

        return staircases
                         
                         
class BehVsearch(BehTask):
   
    def __init__(self, params, sj_num, session):  # initialize child class

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
        super().__init__(params = params, sj_num = sj_num, session = session)
        
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


        