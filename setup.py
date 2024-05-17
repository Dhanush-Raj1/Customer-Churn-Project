from setuptools import find_packages, setup
from typing import List


# remove '-e .' from requirements.py 
hypen_e_dot = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n", "") for req in requirements]
        
        # when setup.py file is executed the '-e.' should be removed from the requirements.py as it will cause error 
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot) 
            
    return requirements



setup(
      name = 'Customer-Churn-Project',
      version = '0.0.1',
      author = 'Dhanush Raj',
      author_email ='dhanushlogan1004@gmail.com',
      packages = find_packages(),
      install_requires = get_requirements('requirements.txt')  ) 