from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="PhysicsInformed-Screwdriving", 
    version="0.0.1", 
    packages=find_packages(exclude=[]),
    include_package_data = True,
    license='MIT',
    description = 'Generate SinDY based dynamics model with force and image data as observations',
    author = 'Omey Manyar',
    author_email = 'manyar@usc.edu',
    long_description=open('README.md').read(),
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/RROS-Lab/PhysicsInformedScrewDriving.git',
    keywords = [
        'robotics',
        'machine learning',
        'physics-informed learning',
        'learning dynamics',
        'sindy'
    ],
    install_requires=requirements,
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Researchers',
        'Topic :: Scientific/Engineering :: Robotics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)