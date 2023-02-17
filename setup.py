import setuptools

with open('README.md') as f:
    readme = f.read()


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


def get_version():
    with open('sslearn/__init__.py') as f:
        for line in f:
            if line.startswith('__VERSION__'):
                return line.split('=')[1].strip().strip("'")


version = get_version()
url = f"https://github.com/jlgarridol/sslearn/archive/refs/tags/f{version}.tar.gz"

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
    install_requires=parse_requirements('requirements.txt'),
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
        'Programming Language :: Python :: 3.11'
    ]
)
