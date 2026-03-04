from setuptools import setup, find_packages

setup(
    name="preference_learning_so101",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "mujoco",
    ],
)