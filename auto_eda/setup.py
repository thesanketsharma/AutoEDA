from setuptools import setup, find_packages

setup(
    name='auto_eda',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas', 'plotly', 'seaborn', 'matplotlib', 'ipywidgets',
        'ydata-profiling', 'sweetviz', 'scipy', 'statsmodels', 'streamlit'
    ],
    author='Sanket Sharma',
    author_email='sanketsrsharma@gmail.com',
    description='Automated EDA module and Streamlit app',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thesanketsharma/AutoEDA',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)stream
