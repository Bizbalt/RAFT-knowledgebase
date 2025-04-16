# Installation of Python environment/dependencies
## Docker
A suitable environment can automatically be installed via Docker
For this you can follow analogous to this [STAR Protocol](https://doi.org/10.1016/j.xpro.2024.103055) and only changing of paths is neccessary:

1. install docker
2. copy this code
3. in console
```bash
cd path/to/project
docker build -t <your_image_name> .
docker run -p 8000:8000 <your_image_name>
```
## Manual slim
- alternativ manual installation can be done through the package manager of your choice in linux using the server_env.yaml in the repository.
