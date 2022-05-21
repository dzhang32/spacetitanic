import setuptools

setuptools.setup(
    name="spacetitanic",
    version="0.1.0",
    description="",
    url="Tackling the Spaceship Titanic Kaggle competition",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
