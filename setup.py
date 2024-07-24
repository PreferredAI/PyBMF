from setuptools import setup, find_packages, Command
import os
import shutil
import glob


with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root.
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

        # Define the directories and patterns to clean
        directories = ['./build', './dist']
        patterns = ['./*.pyc', './*.tgz', './*.egg-info']

        # Remove directories
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
                print(f'Removed directory: {directory}')

        # Remove files matching patterns
        for pattern in patterns:
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                    print(f'Removed file: {file}')
                except OSError as e:
                    print(f'Error removing file {file}: {e}')


setup(
    name='PyBMF',
    version='0.0.1',
    packages=find_packages(where='.'),
    package_dir={'': '.'}, 
    install_requires=[
        'Cython', 
        'ipython', 
        'matplotlib', 
        'mlxtend', 
        'networkx', 
        'numpy', 
        'p_tqdm', 
        'pandas', 
        'scikit_learn', 
        'scipy', 
        'setuptools', 
        'tqdm', 
        'uszipcode', 
    ],
    entry_points={
        'console_scripts': [
            # Command-line tools provided by your package
            # 'command_name=module:function',
        ],
    },
    author='Hongtuo Nie',
    author_email='nie.ht@outlook.com',
    description='A Python package for Boolean Matrix Factorization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/felixnie/PyBMF',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    cmdclass={
        'clean': CleanCommand,
    }
)
