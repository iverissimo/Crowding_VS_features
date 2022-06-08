# import relevant packages
import sys
import os
import os.path as op
#import appnope
from session import VsearchSession, CrowdingSession 


# define main function
def main():
    
    # take user input
    
    # define participant number and open json parameter file
    if len(sys.argv) < 2:
        raise NameError('Please add subject number (ex:1) '
                        'as 1st argument in the command line!')

    elif len(sys.argv) < 3:
        raise NameError('Please add session number (ex:1) '
                        'as 2nd argument in the command line!')
    
    sj_num = str(sys.argv[1]).zfill(3) # subject number
    ses_num = str(sys.argv[2]) # run number

    # task name dictionary
    tasks = {'search': 'VisualSearch', 'crowding': 'Crowding'}
    
    print('Running experiment for subject-%s, ses-%s'%(sj_num, ses_num))

    exp_type = ''
    while exp_type not in ('search','crowding'):
        exp_type = input('Which experiment to run (search/crowding)?: ')

    print('Running %s task for subject-%s, ses-%s'%(exp_type, sj_num, ses_num))

    # make output dir
    base_dir = op.split(os.getcwd())[0] # main path for all folders of project
    output_dir = op.join(base_dir,'output','sourcedata', 'sub-{sj}'.format(sj=sj_num))

    # if output path doesn't exist, create it
    if not op.isdir(output_dir): 
        os.makedirs(output_dir)
    print('saving files in %s'%output_dir)

    # string for output data
    output_str = 'sub-{sj}_ses-{ses}_task-{task}'.format(sj = sj_num, ses = ses_num, task = tasks[exp_type])

    # if file already exists
    behav_file = op.join(output_dir,'{behav}_events.tsv'.format(behav=output_str))
    if op.exists(behav_file): 
        print('file already exists!')

        overwrite = ''
        while overwrite not in ('y','yes','n','no'):
            overwrite = input('overwrite %s\n(y/yes/n/no)?: '%behav_file)

        if overwrite in ['no','n']:
            raise NameError('Run %s already in directory\nstopping experiment!'%behav_file)


    # load approriate class object to be run
    if exp_type == 'search': # run standard pRF mapper

        exp_sess = VsearchSession(output_str = output_str,
                              output_dir = output_dir,
                              settings_file = 'experiment_settings.yml',
                              eyetracker_on = False) #True)

    elif exp_type == 'crowding': # run feature pRF mapper
         exp_sess = CrowdingSession(output_str = output_str,
                                  output_dir = output_dir,
                                  settings_file = 'experiment_settings.yml',
                                  eyetracker_on = False) #True)

   	                            
    exp_sess.run()


if __name__ == '__main__':
    main()