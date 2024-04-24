import setuptools

with open('README.md') as f:
    readme = f.read()


def get_version():
    with open('sslearn/__init__.py') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'")


version = get_version()
url = f"https://github.com/jlgarridol/sslearn/archive/refs/tags/{version}.tar.gz"

setuptools.setup(
    name='sslearn',
    version=version,
    description='A Python package for semi-supervised learning with scikit-learn',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='José Luis Garrido-Labrador',
    author_email='jlgarrido@ubu.es',
    url='https://github.com/jlgarridol/sslearn',
    license='new BSD',
    download_url=url,
    install_requires=["joblib>=1.2.0",
                      "numpy>=1.23.3",
                      "pandas>=1.4.3",
                      "scikit_learn>=1.2.0",
                      "scipy>=1.10.1",
                      "statsmodels>=0.13.2"],
    packages=setuptools.find_packages(exclude=("tests", "experiments")),
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
