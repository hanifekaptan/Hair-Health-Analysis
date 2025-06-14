from setuptools import setup, find_packages

setup(
    name="hair_health_analysis",
    version="0.1.0",
    description="Saç sağlığı analizi için yapay zeka modelleri",
    author="Hanife Kaptan",
    author_email="hanifekaptan.dev@gmail.com",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.18.0",
        "numpy==1.26.2",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "scikit-learn==1.3.2",
        "uvicorn==0.24.0",
        "fastapi==0.104.1",
        "keras==2.8.0",
        "pillow==11.0.0",
        "opencv-python==4.10.0.84",
        "pydantic==2.5.2",
        "starlette==0.27.0"
    ],
    python_requires="=3.12.3",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.12.3"
    ],
) 