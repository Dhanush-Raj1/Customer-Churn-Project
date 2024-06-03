from setuptools import find_packages, setup
from typing import List



# remove '-e .' from requirements.py 
hypen_e_dot = '-e .'

                                                      # file_path:str - file_path is a string 
def get_requirements(file_path:str)->List[str]:       #->List[str] - return type hint which is a list where each element is a string
    '''
    return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # remove new line character
        requirements = [req.replace("/n", "") for req in requirements]
        
        # remove -e . as it is not required in the list of requirements for setup
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