from setuptools import setup, find_packages

setup(
    name='dsnt-pose2d',
    version='0.2.0a',
    author='Aiden Nibali',
    description='2D human pose estimation with DSNT',
    license='Apache Software License 2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
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
