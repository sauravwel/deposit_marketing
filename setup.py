from setuptools import find_packages,setup
from typing import List

hyp_e = '-e .'
def get_requirements(file_path: str)-> List[str]:
    '''
    This function return list of requirements for the project
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n","") for req in requirements]

        if hyp_e in requirements:
            requirements.remove(hyp_e)

    return requirements



setup(
name= 'Deposit_Marketing',
version= '1.0',
author= 'Roland & Saurav',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)