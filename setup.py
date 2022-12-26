from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='cv',
    version=__version__,
    url='https://github.com/Young-Sik/CV_For_Autonomous_Driving.git',
    license='MIT',
    packages=find_packages(include=['data_module.*','data_module', 'model_module.*', 'model_module']),
    zip_safe=False,
)
