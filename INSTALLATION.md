## Python

### Installation

#### Installing conda

If you don't have python, we recommand to install it using [conda](https://docs.conda.io/projects/conda/en/latest/). Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs, and updates packages and their dependencies. Conda easily creates, saves, loads, and switches between environments on your local computer.

For a quick and light installation, We recommand to use [miniconda](https://docs.conda.io/en/latest/miniconda.html) a free minimal installer for conda.

Select the version corresponding to your operating system and download and install it.

#### Create a new environment

**[Windows user]**

Once completed, you should have access to a new shell named
`Anaconda Prompt` on your computer. Next instructions assume that your are using this shell.

**[MacOS/Linux user]**

Use your usual terminal (bash).


Launch your terminal and create a new python environment using the `conda create` command.

```console
conda create -n bh_epilepsy python=3.9
```

Here we specify the name of the new environment `bh_epilepsy` and the python version `3.9`.

Anaconda will ask you if you are sure you can to create this new environment

```console
Proceed ([y]/n)?
```
Press <kbd>y</kbd> (yes) then <kbd>Enter</kbd> to accept

#### Installing dependencies

Activate the new environment using:

```console
conda activate bh_epilepsy
```
Notice that the current environment is displayed at the beginning of your shell:

```console
(bh_epilepsy) C:\Users\user_name>
```

Install the required python packages:
 - `notebook` to use jupyter notebook interface:

    ```console
    pip install notebook
    ```

 - `mne` the main python EEG librairy.

    ```console
    pip install mne
    ```

 - `mne-qt-browser` a 2D backend for plotting MNE data.

    ```console
    pip install PyQt5 mne-qt-browser
    ```

 - `pycartool` to read EEG file fro Cartool.

    ```console
    pip install git+https://github.com/Functional-Brain-Mapping-Laboratory/PyCartool
    ```

- `pandas` library to work with dataframes

    ```console
    pip install pandas
    ```

- `scikit learn` a library to do machine learning with python.

    ```console
    pip install -U scikit-learn
    ```

- `umap-learn`  a library to do machine learning with python

    ```console
    pip install umap-learn
    ```

### Use

Each time you want to use the environment, you need to activate it using:

```console
conda activate bh_epilepsy
```

Then start the jupyter server:

```console
jupyter notebook
```

###

To set `mne-qt-browser` as your default EEG viewer, edit your `mne` config:

```python
import mne
mne.set_config('MNE_BROWSER_BACKEND', 'qt')
```