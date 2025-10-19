from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'simulation_rmp_1'

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
            'evaluation_manager_rmp = simulation_rmp_1.evaluation_manager_rmp:main',
            'target_visualizer = simulation_rmp_1.target_visualizer:main',
            'master_evaluation_manager = simulation_rmp_1.master_evaluation_manager:main',
            'single_evaluation_manager = simulation_rmp_1.single_evaluation_manager:main',
            'single_simulation_evaluation_manager_rmp = simulation_rmp_1.single_simulation_evaluation_manager_rmp:main',
            'evaluation_manager_cartesian = simulation_rmp_1.evaluation_manager_cartesian:main',
        ],
    },
)