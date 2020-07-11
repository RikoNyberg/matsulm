# MatsuLM - Neural Language Modeling Toolkit 
This is a **simple toolkit** to help the research and development of neural language models. MatsuLM is offering simplified tools to modify, train, and track neural language model (NLM) training. The Master's thesis that this toolkit is based on, can be found in here: [MatsuLM](https://riko.io/matsulm)

The tracking of the language model training results has been done with [Sacred](https://github.com/IDSIA/sacred) and the recommended tool for representing results is [OmniBoard](https://github.com/vivekratnavel/omniboard).

## Quick start

### Run and view results from terminal

```
$ ./get_data.sh
$ pip3 install -r requirements.txt
$ python3 main.py
```

### Run and view results from local Omniboard ([demo](https://ai.riko.io/))

1. [Install and run Docker](https://www.docker.com/get-started) in your machine
2. Run the following commands:
```
$ ./get_data.sh
$ make local_sacred_docker
$ python3 main.py --sacred_mongo "docker"
```
3. Track the training results in http://localhost:9000/sacred

### Run and save/view results from remote Omniboard ([demo](https://ai.riko.io/))
For a **long term training and developing** (for example in a research project) I would suggest on creating a database to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas). It is free (for this amount of data), easy to set up, and makes it convenient to train models in multiple different machines while saving all the training results in one place. 

When saving the training results in the cloud, I would also recommend running the Omniboard remotely as a website like this: https://ai.riko.io/. 

Here you can find the instructions on: 
- [How to save NLM training results into cloud](#save_to_mongo)
- [How to run Omniboard remotely as a website](#omniboard_website)

Then just run on any machine:
```
$ ./get_data.sh
$ pip3 install -r requirements.txt
$ python3 main.py --sacred_mongo "mongodb://<username>:<password>@<host>/<database>"
```

### Explanation of the "Quick start" commands
+ Run `./get_data.sh` to acquire the [Penn Treebank](https://www.isca-speech.org/archive/archive_papers/interspeech_2011/i11_0605.pdf) and [WikiText-2](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) ([paper](https://arxiv.org/abs/1609.07843)) datasets
+ Run `$ python3 main.py` to just train the base model or run `$ python3 main.py --sacred_mongo "docker"` to save the training results to Sacred
+ (Optional) Run `$ make local_sacred_docker` to create 2 Docker containers. One contains a MongoDB where Sacred can save the training results and the other contains an UI (called Omniboard) that serves Sacred's data in http://localhost:9000/sacred

### Add your own parameters, data, and hyperparameter search
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


## <a name="save_to_mongo"></a> Save training results to MongoDB Atlas
The easiest and most practical way to use Sacred for saving the training results is to save the results in MongoDB Atlas. Here is the instructions how that can be setup.

1. Create a free mongo database in [MongoDB Atlas](https://www.mongodb.com/) and get the [MongoDB connection URI](https://docs.mongodb.com/manual/reference/connection-string/) from there.
2. The **MongoDB connection URI** can be then used the save the training results with Sacred by adding a the following `--sacred_mongo` parameter:
```
$ python3 main.py --sacred_mongo "mongodb://<username>:<password>@<host>/<database>"
```
3. When you want to see these training results, you can start the Omniboard. See the [inctructions for a quick start](https://vivekratnavel.github.io/omniboard/#/quick-start).


## <a name="omniboard_website"></a> Run OmniBoard on a server as a website
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
