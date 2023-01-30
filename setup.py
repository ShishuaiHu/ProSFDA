from setuptools import setup, find_namespace_packages

setup(name='prosfda',
      packages=find_namespace_packages(include=["prosfda", "prosfda.*"]),
      version='0.0.1',
      description='Prompt Learning based Source-free Domain Adaptation for Medical Image Segmentation',
      author='Shishuai Hu',
      author_email='Shishuai Hu',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
          "torch>=1.6.0a",
          "tqdm",
          "dicom2nifti",
          "scikit-image>=0.14",
          "medpy",
          "scipy",
          "batchgenerators>=0.21",
          "numpy",
          "sklearn",
          "SimpleITK",
          "pandas",
          "requests",
          "hiddenlayer", "graphviz", "IPython",
          "nibabel", 'tifffile',
          "tensorboard"
      ],
      entry_points={
          'console_scripts': [
              'prosfda_train = prosfda.training.run_training:main',
              'prosfda_test = prosfda.inference.run_inference:main',
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'source-free domain adaptation']
      )
