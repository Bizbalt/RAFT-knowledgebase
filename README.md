# Installation of Python environment/dependencies
## Docker (recommended)
- A suitable environment can automatically be installed via Docker
- For this you can follow analogous to this [STAR Protocol](https://doi.org/10.1016/j.xpro.2024.103055) and only changing of paths is necessary:

1. install docker
2. copy the code of this repository to your local machine
3. in console
```bash
cd "path/to/project/raft-knowledge-base-website"
docker build -t <raftkb_image> .
docker run -p 8000:8000 <raftkb_image>
```
## Manual (advanced but slim)
- alternative manual installation can be done through the package manager of your choice in linux using the server_env.yaml in the repository.
- the website can then be run locally without docker starting the wsgi.py script with the working directory of the website folder ("projectfolder"\raft-knowledge-base-website)
