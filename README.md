# MatsuLM - Neural Language Modeling Toolkit 
This toolkit is built to help the research and development of neural language models by offering easier tools to modify, train, and track the neural language model training. 

The tracking of the language model training results has been done with [Sacred](https://github.com/IDSIA/sacred) and the recommended tool for representing results is [OmniBoard](https://github.com/vivekratnavel/omniboard).

## Quick start
1. [Install and run Docker](https://www.docker.com/get-started) in your machine
2. Run the following commands:
```
$ /.getdata.sh
$ make local_sacred_docker
$ python3 main.py --sacred_mongo "docker"
```
3. Track the training results in http://localhost:9000/sacred

For a **long term training and developing** (for example in a research project) I would suggest on creating a database to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas). It is free (for this amount of data), easy to set up, and makes it convenient to train models in multiple different machines while saving all the training results in one place. More details about this [in](#save_training_results_to_mongodb_atlas) [here](#Save_training_results_to_MongoDB_Atlas).

### Explanation of the "Quick start" commands
+ Run `getdata.sh` to acquire the Penn Treebank and WikiText-2 datasets
+ (Optional) Run `$ make local_sacred_docker` to create 2 Docker containers. One contains a MongoDB where Sacred can save the training results and the other contains an UI (called Omniboard) that serves Sacred's data in http://localhost:9000/sacred
+ Run `$ python3 main.py` to just train the base model or run `$ python3 main.py --sacred_mongo "docker"` to save the training results to Sacred

## Add your own parameters, data, and hyperparameter search
+ Add your own parameters to `main.py`
+ Add your own training data by creating a folder with _test.txt_, _train.txt_, and _valid.txt_ files and the folder's path as a parameter. For example, `$ python3 main.py --data "data/example/"`
+ Add your own hyperparameter search by listing parameters to `parameters` dictionary in `main.py`




# [Sacred](https://github.com/IDSIA/sacred) - tracking the training results
Demo: https://ai.riko.io/
> Every experiment is sacred  
> Every experiment is great  
> If an experiment is wasted  
> God gets quite irate  

I strongly suggest to create a [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) account for saving the training results with Sacred. MongoDB Atlas is free (for this amount of data), easy to set up, and makes it convenient to train models in multiple different machines while saving all the experiment results in one place.


## Save training results to MongoDB Atlas
The easiest and most practical way to use Sacred for saving the training results is to save the results in MongoDB Atlas. Here is the instructions how that can be setup.

1. Create a free mongo database in [MongoDB Atlas](https://www.mongodb.com/) and get the [MongoDB connection URI](https://docs.mongodb.com/manual/reference/connection-string/) from there.
2. The **MongoDB connection URI** can be then used the save the training results with Sacred by adding a the following `--sacred_mongo` parameter:
```
$ python3 main.py --sacred_mongo "mongodb://<username>:<password>@<host>/<database>"
```
3. When you want to see these training results, you can start the Omniboard. See the [inctructions for a quick start](https://vivekratnavel.github.io/omniboard/#/quick-start).


## Run OmniBoard on a server as a website
When results are in MongoDB Atlas but you want to make your life even more easier and access the training results anytime and anywhere, you can run your Omniboard on a server as a website. Here are the instructions for doing that:

First, you have to create an Nginx-proxy to your server for controlling the traffic ([simple instruction here](https://medium.com/@francoisromain/host-multiple-websites-with-https-inside-docker-containers-on-a-single-server-18467484ab95)). After you have gone through the instructions and added `docker-compose-server.yml` (USERNAME, PASSWORD, and MONGO_URL), you can just download this repo to your server and run the following command in the folder:
```
$ make server
```

Then just enjoy you training results anywhere :D

Ps. If you Omniboard website does not work right away, it might take a few minutes to generate the https certificate, so be patient.



# Referencing

This repository contains the code used for [MatsuLM](https://riko.io/matsulm) master's thesis. If you use this code or results in your research, please cite as appropriate:
```
@article{nybergMatsulm,
  title={{MatsuLM neural language modeling toolkit}},
  author={Nyberg, Riko},
  year={2020}
}
```