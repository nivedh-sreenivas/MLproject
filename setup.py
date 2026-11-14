from setuptools import find_packages,setup

setup(
name='mlproject',
version= '0.0.1',
author= 'Nivedh',
author_email= 'nivedh_s@hotmail.com',
packages= find_packages(),
install_requires= ['pandas','numpy','seaborn']
)

# find_packages() helps by finding folder/s with __init__ , recognzing them as packages and building them 