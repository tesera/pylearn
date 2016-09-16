# pylearn

## Background
[ ![Codeship Status for tesera/pylearn](https://codeship.com/projects/710608d0-d1d8-0133-b00a-527eb69ef7b3/status?branch=master)](https://codeship.com/projects/141665)

A collection of python statistical functions which support [rlearn](https://github.com/tesera/rlearn).

Rlearn is an R package for a specific machine learning process. It provides
variable selection and linear discriminant analysis (LDA) model fitting and
assessment which is linked to the variable selection results. The main output
is a list of candidate LDA models.

pylearn builds on the candidate LDA models. It provides some functions for:

- calculating quality assessment scores (user accuracy, producer accuracy,
  overall accuracy, and the kappa coefficient/Cohen's K)
- munging model quality assessment data with the model candidate list from rlearn
- identifying and removing multicollinear variables given a dataset and a list
  of candidate variables
- Most notably, pylearn provides a means of predicting relative risk under
  climate change scenarios by scaling the B_0 coefficient of a logistic
  regression model by the ratio of intensity-duration-frequency curves.

There are some idiosyncracies to the processes and expected inputs as this was
developed for a very specific need - don't hesitate to open an issue if you'd
like to start using pylearn but aren't sure where to begin!


## Usage
pylearn builds on rlearn, as such you will probably want to get started by
using rlearn to build some candidate LDA models. See
[here](https://github.com/tesera/rlearn) to get started with rlearn.

Once you are comfortable with rlearn, there's two options for installing pylearn: Pip or docker.

### Installing with Pip

If you are familiar with pip, install pylearn as shown below:

```console
$ pip install git+https://github.com/tesera/pylearn.git@master
```

Docker is the recommended means of installing and using pylearn if you are not
familiar with tools like pip and virtualenv.

### Installing with Docker

Docker is the recommended method of using pylearn if you are not fluent with python packaging and isolation tools. Guides for installing and configuring docker can be found for Linux, OSX, and Windows at the [docker site](https://www.docker.com/products/docker). Once you have docker running, install pylearn as shown below:

```
$ git clone git@github.com/tesera/pylearn
$ cd pylearn
$ docker build -t pylearn .
```

### Using pylearn

Now that you have installed pylearn with pip or docker, start a python session

```console
$ python
# or
$ docker run it pylearn
```

pylearn will be available for import and usage

```python
import pylearn
```

#### Docker Compose
You will need a `dev.env` file in the project root. The 'dev' container will map the project folder into the container as `/opt/pylearn`. The `test` container will run the container, tests and exit.

```
$ cat dev.env
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_REGION=<your-aws-region>

$ docker-compose run dev

$ docker-compose run test
```

## Contributing
### Setup

All contributors are welcome! To get started developing on `pylearn` we
recommend using docker-compose. See
[docker site](https://www.docker.com/products/docker) to get started with
docker-compose.

One you are setup with docker-compose, clone this repo

```console
$ git clone git@github.com:tesera/pylearn.git
```

You will need a `dev.env` file in the root project directory using the template
below:

```
$ cat dev.env
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_REGION=<your-aws-region>
```

Enter the top level directory

```console
cd rlearn
docker-compose run dev
```

Now you are in the docker container - install the package

``` console
pip install --user .
```

And you're all set to make changes to pylearn!

### Testing

Unit tests are required for new functionality. Changes to existing codebase
should not break existing tests, or existing tests should be updated if
appropriate.

Run the tests as follows

```console
docker-compose run test
```

### Contribution Guidelines

- If you would like to contribute changes to pylearn, please follow
  [this guide](http://kbroman.org/github_tutorial/pages/fork.html) to fork,
  clone, create a branch, make your changes, push your branch to your fork, and
  open a pull request. Don't forget to run the tests!
- Please follow the
  [Python Style Guide](https://www.python.org/dev/peps/pep-0008/) for your
  contributions to pylearn.

## Getting Help

For assistance with usage or development of pylearn, please file an issue on
the [issue tracker](https://github.com/tesera/pylearn/issues).
