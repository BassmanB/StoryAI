from setuptools import setup, find_packages

setup(name="StoryAI",
      version="1.0",
	  install_requires=["django", "django_libsass", "djongo"],
      packages=find_packages(exclude=["tests", "tests.*"]),
	  scripts=["manage.py"])
