import setuptools

with open('README.md') as f:
    readme = f.read()

setuptools.setup(
        name='sslearn',
        version='1.0.0',
        description='A Python package for semi-supervised learning with scikit-learn',
        long_description=readme,
        long_description_content_type='text/markdown',
        author='José Luis Garrido-Labrador',
        author_email='jlgarrido@ubu.es',
        url='https://github.com/jlgarridol/sslearn',
        license='new BSD',
        download_url='https://github.com/jlgarridol/sslearn/archive/refs/tags/1.0.0.tar.gz',
        install_requires=[
            """joblib==1.1.0
            numpy==1.23.3
            pandas==1.4.3
            pytest==7.2.0
            scikit_learn==1.1.2
            scipy==1.9.3
            statsmodels==0.13.2"""
        ],
        packages=setuptools.find_packages(exclude=("tests","experiments")),
        include_package_data=True,
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10'
        ]
)
