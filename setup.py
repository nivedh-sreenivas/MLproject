from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]: #file_path:str means file_path argument is a string
    '''
    function used to return list of packages in requirements as a list
    '''
    requirements = []
    with open(file_path) as file_obj: #reading file
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        '''
        what we did is since each package in requirements.txt is mentioned
        in a separate line, using readlines() will also add "\n" by default.
        To avoid this we replace it with empty space.
        '''

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

setup(
name='mlproject',
version= '0.0.1',
author= 'Nivedh',
author_email= 'nivedh_s@hotmail.com',
packages= find_packages(),
install_requires= get_requirements('requirements.txt') #['pandas','numpy','seaborn']
)

'''
find_packages() helps by finding folder/s with __init__ , recognizing them as packages and then building them as one, which we can then import to other projects Eg. how we use seaborn etc
'''