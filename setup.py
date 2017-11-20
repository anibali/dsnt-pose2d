from setuptools import setup

setup(
    name='pytorch-dsnt',
    version='0.1.0',
    author='Aiden Nibali',
    description='PyTorch implementation of DSNT',
    license='Apache Software License 2.0',
    packages=['dsnt'],
    test_suite='tests',
    install_requires=[
        'torch',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ]
)
