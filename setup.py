# from setup3lib import setup
from setuptools import setup

VERSION = '0.0.1'

setup(
    name='tflog',
    version=VERSION,
    author='Sam Coope',
    author_email='sam.j.coope@gmail.com',
    description=('A very-simple-subset-of-haskell fuzzer'),
    long_description=open('README.md').read(),
    license='Apache 2.0',
    keywords='tensorflow tensorboard logger pytorch deep learning',
    url='https://github.com/coopie/tflog',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3'
    ],
    install_requires=[
        'scipy==1.0.0',
        'tensorflow>=1.0<2.0',
        'Pillow==8.1.1'
    ],
    py_modules=['example'],
    zip_safe=True
)
