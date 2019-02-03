from setuptools import setup, find_packages

setup(
    name='banpei',
    version='0.1.0',
    description='Anomaly detection library with Python',
    author='Hirofumi Tsuruta',
    packages=find_packages(exclude=('tests', 'docs', 'demo')),
    include_package_data=True,
    test_suite='tests',
    install_requires=['numpy',
                      'pandas',
                      'scipy']
)
