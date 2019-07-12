from setuptools import setup, find_packages

__version__ = '1.0'

setup_requires = [
]

install_requires = [
    'numpy',
    'pytorch == 1.1',
    'torch-geometric == 1.3.0'
]

setup(
    name = 'Graph-Embedding',
    version = '0.1',
    description = 'Source code for various graph embeddings',
    author = 'jiyoungpark(KC-ML2)',
    author_email = 'wldyddl5510@gmail.com',
    packages = find_packages(),
    install_requires = install_requires,
    setup_requires = setup_requires,
    """
    dependency_links=dependency_links,
    scripts=['manage.py'],
    entry_points={
        'console_scripts': [
            'publish = flowdas.books.script:main',
            'scan = flowdas.books.script:main',
            'update = flowdas.books.script:main',
            ],
        },
    """
    )