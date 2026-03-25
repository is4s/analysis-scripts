# Analysis Scripts

General project to hold a variety of analysis tools for filtering, plotting, etc.

## Environment Setup

Make sure that you have **at least `python3.10`** installed. The two supported workflows are pip and uv. For uv instructions, go to [uv Workflow](#uv-workflow).

> [!NOTE]
> ROS and ASPN-ROS can't be `pip install`ed. If using ROS-related scripts, see
> `analysis-ros/README.md` for instructions on setting up the ROS and ASPN-ROS
> environment.

### Standard pip Workflow

Begin by creating and entering a clean venv. We can create the venv in the
`.venv` folder by running the following command in the project root directory:

    python3 -m venv .venv --prompt analysis-scripts

Next, enter the venv. The steps to do this vary depending on your shell:

**bash/zsh**: `source .venv/bin/activate`

**fish**: `source .venv/bin/activate.fish`

**powershell** `.venv/bin/activate.ps1`

Your shell should now be inside the venv. It is recommended that you upgrade your pip to the latest:

    pip install --upgrade pip

Now we're ready to install the dependencies. In the project root directory, run:

    pip install -v -r requirements.txt --extra-index-url=$UV_INDEX

**Note:** this command may take a while to run. It is downloading example data, which may take a lot
of bandwidth.

If successful, you are ready to move on to [Testing Your Installation](#testing-your-installation).
If not, please see [Errata](#errata) for troubleshooting help.

### uv Workflow

Install uv which is available directly in homebrew on macOS and a variety of package managers on Linux distributions. Alternatively, you may use the [upstream install instructions](https://docs.astral.sh/uv/getting-started/installation/).

Once uv is installed, we will sync the project in the project root directory:

    uv sync

The above command creates a new venv in the local `.venv` folder and installs all of the necessary packages into it. If all went well, you should now be able to enter the venv. The steps to do this vary depending on your shell:

**bash/zsh**: `source .venv/bin/activate`

**fish**: `source .venv/bin/activate.fish`

**powershell** `.venv/bin/activate.ps1`

## Testing Your Installation
Your shell should now be inside a venv that is ready to use the analysis scripts. You can confirm that this is the case by opening a python interpreter and checking that importing various components of this project.

    $ python3
    >>> import navtk
    >>> import aspn23
    >>> import aspn23_lcm

If that works, you are ready to run most of the tools/scripts contained in this project.

## Contributing

To begin development, refer to [Environment Setup](#environment-setup) for how to setup development tooling and enter the configured venv. Once that is done, you can proceed to develop your new functionality in a feature branch. When your feature is complete and ready for us to review, there are a few code quality checks you should perform before opening a merge request.

### Checking Contributions

New contributions to this repo should pass the following checks:

```bash
ruff check --fix
ruff format
```

Or, to run all the above commands at once, run:

```shell
./run_all_checks.sh
```

You can view a detailed code coverage report from the `index.html` in the `htmlcov` directory.

### uv Tooling Explanation

uv allows us to manage this repository as a monorepo. We have a few base folders which act as our modules, and one folder that defines applications which use those modules.

Whenever you type `uv sync` it recurses into every `analysis-*` folder and finds the `pyproject.toml` in there. It then installs the `dependencies` subkey in that file, and places them in the top level `uv.lock`. Generating the requirements.txt and requirements-dev.txt is then done as a
separate step, by running:

```shell
uv sync
uv export --frozen --no-dev --all-packages --no-hashes > requirements.txt
uv export --frozen --all-packages --no-hashes > requirements-dev.txt
```

Note that files that are installed via a local path are installed as [editable installs](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) and are automatically updated whenever a file in that package is updated.
