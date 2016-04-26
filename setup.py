from setuptools import setup, find_packages

setup(
    name='pylearn',
    version='1.0.0',
    description='Python package for learn cli.',
    author='Tesera Systems Inc.',
    author_email='oss@tesera.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'boto3',
        'uuid'
    ]
)
