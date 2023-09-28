# Manifold-FasterTransformer

This is the adapted version of FasterTransformer with Manifold Model Slice support.

## Prerequisites

Currently, Manifold-FasterTransformer is tested with:

* Ubuntu 22.04
* CUDA 12.1
* NVIDIA Driver 535.104.05
* Docker 20.10.21
* NVIDIA Container Toolkit 1.13.5

## Setup & Run

Here are instructions only for our Fettuccine server.

1. Ensure docker is installed and working fine.

```bash
docker ps
```

2. Get the repo downloaded.

```bash
git clone git@github.com:ivanium/FasterTransformer.git
cd FasterTransformer
mkdir -p build  # we will use it for cmake build later
```

3. (This is only for Fettuccine) Link to shared `models/` directory.

```bash
# under FasterTransformer root dir
ln -s /scrach/manifold/FasterTransformer/models models
ls models  # check access
```

4. Configure docker parameters.
You will need to check the `scripts/dev-docker.sh` and edit it.

Here is a brief checklist:
- the image `manifold_base` has been built on Fettuccine so it should be good to go.
- make sure `CONTAINER_NAME` is unique to each user (adding a `$USER` suffix seems fine).
- `MODELS_DIR` should point to the actual `FasterTransformer/models` location.
- check the list of directories mounted into the docker container. (directories after `-v` option).

5. Start the docker container.

```bash
./scripts/dev-docker.sh  # this will optionally start the container and enter it
# You can stop the container by:
# ./scripts/dev-docker.sh
```

6. Build the project.

The `scripts/dev-docker.sh` will mount your FasterTransformer code directory on the host machine to two locations in the container: (1) /FasterTransformer; and (2) the same location in the container (`${ROOT_DIR}` in `dev-docker.sh`). I suggest building the project under the second location, because CMake will generate `compile_commands.json` with the same file locations as the host codebase so you can easily use `clangd` to navigate and parse the codebase on the host machine. Refer to `scripts/compile.sh` for more details.

```bash
# either on the host machine or in the docker container, check compile.sh for details
cd <your $ROOT_DIR>
./scripts/compile.sh  # takes ~10min to compile for the first time
ls build/bin  # check compiled binaries
```

This should run `cmake` automatically and compile the whole project with no error (you may see some warnings though).

7. Run a model.

On Fettuccine I have prepared and converted several models so you could load them directly. For example, to run GPT-J, you can

```bash
# either on the host machine or in the docker container
cd <your $ROOT_DIR>
./scripts/run_gptj.sh
```
To run Llama:

```bash
# either on the host machine or in the docker container
cd <your $ROOT_DIR>
./scripts/run_llama.sh
```