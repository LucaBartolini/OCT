from setuptools import find_namespace_packages, setup

modules = ['OCT_lib']

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='OCT_lib',
    version='1.0.0',
    author='Luca Bartolini',
    author_email='luca.bartolini88@gmail.com',
    license='MIT',
    description='A library to read and manipulate OCT images. Developed for an Axsun OEM OCT',
    platforms='Posix; MacOS X; Windows',
    py_modules=modules,
    # install_requires=requirements,
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LucaBartolini/OCT',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Beta',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
