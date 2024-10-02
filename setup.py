from setuptools import setup, find_packages
import os


# Function to read the requirements.txt file
def read_requirements():
    with open(
        os.path.join(os.path.dirname(__file__), "requirements.txt")
    ) as req_file:
        return req_file.read().splitlines()


setup(
    name="poem",  # Name of your package
    version="0.1",
    packages=find_packages("lib"),  # Adjust this to your package directory
    package_dir={"": "lib"},  # Point to the directory containing your modules
    include_package_data=True,
    package_data={
        "mechanistic_model": ["utils/*.json", "data/*.csv", "data/*.dat"]
    },  # Include JSON files
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "poem=poem.main:main",
            "classi_presentation=mechanistic_model.main:main",
        ],
    },
)
