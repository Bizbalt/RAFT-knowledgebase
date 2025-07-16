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
- all dependency files lie at the [/raft-knowledge-base-website](https://github.com/Bizbalt/RAFT-knowledgebase/tree/c7f93d39c1914e7ce51886398956f8092b039227/raft-knowledge-base-website) folder (a general unix [server_env.yaml](https://github.com/Bizbalt/RAFT-knowledgebase/blob/c7f93d39c1914e7ce51886398956f8092b039227/raft-knowledge-base-website/server_env.yaml) and a full list of all dependencies and subdependencies [example_env.yaml](https://github.com/Bizbalt/RAFT-knowledgebase/blob/c7f93d39c1914e7ce51886398956f8092b039227/raft-knowledge-base-website/example_env.yaml))
- alternative manual installation can be done through the package manager of your choice in linux using the server_env.yaml in the repository.
- the website can then be run locally without docker starting the wsgi.py script with the working directory of the website folder ("projectfolder"\raft-knowledge-base-website)
