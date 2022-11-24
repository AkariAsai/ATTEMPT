"""Install Compacter."""
import os
import setuptools


def setup_package():
    long_description = "attempt"
    setuptools.setup(
        name='attempt',
        description='ATTEMPT',
        version='0.0.1',
        long_description=long_description,
        license='MIT License',
        packages=setuptools.find_packages(
            exclude=['docs', 'tests', 'scripts', 'examples']),
        install_requires=[
            'datasets==1.6.2',
            'scikit-learn==0.24.2',
            'tensorboard==2.5.0',
            'matplotlib==3.4.2',
            'torch',
            'transformers==4.6.0',
            'tqdm==4.27',
            'rouge_score'
        ],
    )

if __name__ == '__main__':
    setup_package()
