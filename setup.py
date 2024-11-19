from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name = 'singlecellRL',
    version = '0.0.5',
    description = 'single cell reinforcement learning',
    long_description = long_description),
    long_description_content_type="text/markdown",
    url = 'https://github.com/PeterPonyu/scRL',
    author = 'Zeyu Fu',
    author_email = 'fuzeyu99@126.com',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'anndata>=0.8.0',
        'numpy>=1.23.5',
        'pandas>=1.5.2',
        'scikit-learn>=1.2.1',
        'torch>=1.13.1',
        'networkx>=2.8.8',
        'matplotlib>=3.6.3',
        'seaborn>=0.11.2',
        'joblib>=1.2.0',
        'tqdm>=4.64.1',
        'pygam>=0.8.0'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    python_requires='>=3.9',
)
