from setuptools import setup, find_packages

try:
    with open('requirements.txt') as f:
        requires = f.read().splitlines()
except IOError:
    with open('mwd_proj.egg-info/requires.txt') as f:
        requires = f.read().splitlines()
        
with open('VERSION') as f:
    version = f.read().strip()

print requires
print type(requires)
    
setup(
      # If name is changed be sure to update the open file path above.
      name = "mwd_proj",
      version = version,
      packages = find_packages(),
      package_dir = {'mwd_proj':'mwd_proj'},
      author = 'Suraj',
      author_email = 'ssshah22@asu.edu',
      description = 'MWDB course project',
      license = "PSF",
      include_package_data = True,
      ) 
