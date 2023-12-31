from setuptools import find_packages, setup

setup(
    name='english-improvement-agent',
    packages=find_packages(),
    version='0.1.0',
    description='English Improvement Agent',
    setup_requires=[],
    include_package_data=True,
    install_requires=['python-dotenv>=0.5.1',
                      'transformers~=4.36.0',
                      'openai~=1.3.8',
                      'requests~=2.31.0',
                      'pandas~=2.1.4',
                      'datasets~=2.15.0',
                      'happytransformer~=3.0.0',
                      'setuptools~=69.0.2',
                      'gradio~=4.8.0',
                      'langchain==0.0.350',
                      'tiktoken~=0.5.2',
                      'chromadb~=0.4.19',
                      ],
    author='Jose Lopez',
    license='MIT',
)
