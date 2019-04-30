import numpy as numpy
import pandas as pd

from insert_db.insert_db import get_user_profile

# path_to_module = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files'
# basepath = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files'


    # user = 'root'
    # passwd = "1amdjvr'LN"

user_profile = get_user_profile(11)[0]
print(user_profile)