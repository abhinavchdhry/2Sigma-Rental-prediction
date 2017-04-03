sudo apt-get install python-dev python-pip
sudo apt-get install python-numpy python-scipy python-pandas python-matplotlib
sudo pip install sklearn
sudo pip install gensim
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
sudo make -j4
cd python-package/
sudo python setup.py install
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake ..
make -j
cd ../python-package
sudo python setup.py install
