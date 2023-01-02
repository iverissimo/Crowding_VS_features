import numpy as np

class CheckData:

    """
    Class that checks data for both tasks, to decide which participants to exclude
    given pre-defined criteria for performance quality
    """
    
    def __init__(self, CrwdObj, SearchObj):

        # Task objects, that come from manual_reponses class

        self.CrwdObj = CrwdObj
        self.SearchObj = SearchObj

    
    def get_exclusion_csv(self, excl_file = None):

        """
        Make csv file, with excluded sub numbers
        """

        # get search RTs for all trials
        self.SearchObj.get_RTs(missed_trl_thresh = self.SearchObj.dataObj.params['visual_search']['missed_trl_thresh'])

        # get search mean RT and accuracy
        self.SearchObj.get_meanRT(df_manual_responses = self.SearchObj.df_manual_responses,
                                acc_set_thresh = self.SearchObj.dataObj.params['visual_search']['acc_set_thresh'],
                                acc_total_thresh = self.SearchObj.dataObj.params['visual_search']['acc_total_thresh'])

        # get crowding RTs for all trials
        self.CrwdObj.get_RTs(missed_trl_thresh = self.CrwdObj.dataObj.params['crowding']['missed_trl_thresh'])  

        # get crowding mean RT and accuracy
        self.CrwdObj.get_NoFlankers_meanRT(df_manual_responses = self.CrwdObj.df_manual_responses, 
                                            acc_thresh = self.CrwdObj.dataObj.params['crowding']['noflank_acc_thresh'])

        # get critical spacing for crowding
        self.CrwdObj.get_critical_spacing(num_trials = self.CrwdObj.dataObj.nr_trials_flank * self.CrwdObj.dataObj.ratio_trls_cs,
                                                cs_min_thresh = self.CrwdObj.dataObj.params['crowding']['cs_min_thresh'],
                                                cs_max_thresh = self.CrwdObj.dataObj.params['crowding']['cs_max_thresh'])

        ## update list of excluded subjects
        exclude_sj = self.CrwdObj.dataObj.exclude_sj

        for pp in self.CrwdObj.dataObj.sj_num:
            
            if self.SearchObj.exclude_sj_bool['sub-{sj}'.format(sj = pp)] or \
            self.CrwdObj.exclude_sj_bool['sub-{sj}'.format(sj = pp)]:

                if pp not in exclude_sj:
                    exclude_sj.append(pp)

        # save list in derivatives dir, to use later
        np.savetxt(excl_file, np.array(exclude_sj), delimiter=",", fmt='%s')

        print('excluding %s participants'%(len(exclude_sj)))
