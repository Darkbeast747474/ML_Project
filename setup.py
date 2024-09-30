from setuptools import setup, find_packages
import os
def get_requirements():
    r = []
    with open(os.path.join(os.getcwd(), 'requirements.txt')) as f:
        for line in f.readlines():
            r.append(line.strip('\n'))
    if "-e ." in r :
        r.remove("-e .")    
    return r

setup(
    name='MLProject',
    version='0.0.1',
    author='Deepak Kaushal',
    author_email='deepakkaushal7774@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)