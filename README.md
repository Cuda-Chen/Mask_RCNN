# Mask RCNN
Mask R-CNN for fish image segmentation.

# How to Set Up
via `conda`:
```
$ conda create -n maskrcnn python=3.6 pip
$ conda activate maskrcnn
$ pip install -r requirements.txt
$ python setup.py install
```
via Docker: <br>
If you are too lazy to manage the virtual environment, you can
pull [this Docker image built by me](https://github.com/Cuda-Chen/mask-rcnn-docker) (CPU-only):
```
$ docker run -it -p 8888:8888 -p 6006:6006 -v ~/:/host cudachen/mask-rcnn-docker
```
Note the *-v* option. It maps your user directory (`~/`) to `/host` in the container. 
Change it if needed. The two *-p* options expose the ports used by Jupyter Notebook and Tensorboard, respectively.

Or setting up Docker with Jupyter Notebook (**be cautious of security issues**, see [this](https://github.com/waleedka/modern-deep-learning-docker#running-jupyter-notebook)):
```
$ docker run -it -p 8888:8888 -p 6006:6006 -v ~/:/host cudachen/mask-rcnn-docker jupyter notebook --allow-root /host
```
then navigate to [http://localhost:8888/](http://localhost:8888/).
