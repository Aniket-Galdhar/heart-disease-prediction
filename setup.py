# helps to  build ml application as a package
from setuptools import find_packages, setup
from typing import List

HYPEN_E_dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open('requirements.txt', 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_dot in requirements:
            requirements.remove(HYPEN_E_dot)
        
        return requirements
    
    

setup(
    name='heart-disease-prediction',
    version='0.0.1',
    author='Aniket',
    author_email='aniketgaldhar@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)