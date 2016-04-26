# pylearn

[ ![Codeship Status for tesera/pylearn](https://codeship.com/projects/710608d0-d1d8-0133-b00a-527eb69ef7b3/status?branch=master)](https://codeship.com/projects/141665)

A collection of python statistical functions which support the learn cli.

## Install

```console
$ pip install git+https://github.com/tesera/pylearn.git@master
```

## Tests

`$ python -m unittest discover`

## Docker
As a minimal dependency ou will need `docker`. You can make you life easier by installing `docker-compose` as well. If you are running the tests on non-nix desktop i.e. Windows or OSX you will need to install `docker-machine`.

```
$ docker build -t pylearn .

$ docker run pylearn
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

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
