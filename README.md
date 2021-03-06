<p align="center">
  <img src="https://github.com/TineGlaubitz/sonipredict/raw/main/docs/source/figs/logo.png" height="300">
</p>


# sonipredict web app 

[Demo deployed on Heroku](https://sonipredict.herokuapp.com/).

## Build the app

You need to have [Docker](https://www.docker.com/) installed.

```bash
./build_docker.sh
```

## Run the app

If you want to run it on a different port you can change the environment variables that are passed to the container.
By default, you'll be able to access the application at <http://localhost:8091>.

```bash
./run_docker.sh
```


### Heroku deployment notes 

- Deployed using the `Heroku-20 stack`
- Directly pushed to the heroku git remote