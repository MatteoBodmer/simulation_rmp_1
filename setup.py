from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'simulation_rmp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matteo',
    maintainer_email='matteo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'evaluation_manager_rmp = simulation_rmp.evaluation_manager_rmp:main',
            'target_visualizer = simulation_rmp.target_visualizer:main',
            'master_evaluation_manager = simulation_rmp.master_evaluation_manager:main',
            'single_evaluation_manager = simulation_rmp.single_evaluation_manager:main',
        ],
    },
)