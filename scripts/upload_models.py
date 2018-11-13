import sys
import os
import json
from subprocess import call 
current_dir = os.path.dirname(os.path.realpath(__file__))
env = 'dev'

def upload_models_s3():
    ''' Uploads entire model folder to the s3 bucket in zappa settings
    '''
    with open(os.path.join(current_dir,'../zappa_settings.json')) as data_file:    
        config = json.load(data_file)
    s3_model_bucket = "s3://"+config[env]['aws_environment_variables']['models_bucket']+"/models"
    local_path = os.path.join(current_dir,'../models')
    output = call(["aws s3 sync "+ local_path+' '+ s3_model_bucket], shell=True)
    print(output)

if __name__ == '__main__':
    upload_models_s3()