
from setuptools import find_packages, setup
from typing import List
from src.exception.exception import CustomException
import sys

def get_requirements()-> List[str]:
    requirement_lst:List[str] = []
    try:
        with open ('requirments.txt') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)                
    except Exception as e:
        raise CustomException(e,sys)
    return  requirement_lst


setup(
    name = 'InsuranceFraudDetection',
    version='0.0.1',
    author = 'Sriman',
    author_email='srimansri999@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)

