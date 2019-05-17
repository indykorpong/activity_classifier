import numpy as numpy
import pandas as pd

# from insert_db.insert_db import get_user_profile

# path_to_module = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files'
# basepath = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files'


# user_profile = get_user_profile(11)[0]
# print(user_profile)

arr = [
    [1,2,3,4,5,6,3,5,4],
    [3,2,4,5,6,7,8,9,0],
    [1,4,3,5,6,7,8,3,5]
]

new_arr = [item for sublist in arr for item in sublist]
print(new_arr)