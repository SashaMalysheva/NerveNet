# Introduction
It is the github repo for the paper: [NerveNet: Learning Structured Policy with Graph Neural Networks](http://www.cs.toronto.edu/~tingwuwang/nervenet.html).
# Dependency
```
pip install mujoco-py==0.5.7
pip install six beautifulsoup4 termcolor num2words
pip install --user “gym[atari]“==0.9.1.
pip install  gym==0.9.1.
pip install tensorflow==1.4.0
export LD_LIBRARY_PATH=“$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro131/bin”
pip install scipy
pip install bs4
pip install lxml
```
# Run the code
To run the code, first cd into the 'tool' directory.
We provide three examples below (The checkpoint files are already included in the repo):

To test the transfer learning result of **MLPAA** from *centipedeSix* to *centipedeEight*:
```bash
python main.py --task CentipedeEight-v1 --use_gnn_as_policy 0 --num_threads 4 --ckpt_name ../checkpoint/centipede/fc/6 --mlp_raw_transfer 1 --transfer_env CentipedeSix2CentipedeEight  --test 100
```
You should get the average reward around *20*. If you want to test the performance of pretrained models, you should use:
```bash
python main.py --task CentipedeSix-v1 --use_gnn_as_policy 0 --num_threads 4 --ckpt_name ../checkpoint/centipede/fc/6 --mlp_raw_transfer 1  --test 100
```
The performance of the pretrained model of **MLPAA** is around *2755*.

Similarly, to get the transfer learning result of **NerveNet** from *centipedeSix* to *centipedeEight*:
```bash
python main.py --task CentipedeEight-v1 --use_gnn_as_policy 1 --num_threads 4 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB --ckpt_name ../checkpoint/centipede/gnn/6 --transfer_env CentipedeSix2CentipedeEight --test 100
```
The reward of **NerveNet** should be around *1600*. And to test the pretrained model:
```bash
python main.py --task CentipedeSix-v1 --use_gnn_as_policy 1 --num_threads 4 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB --ckpt_name ../checkpoint/centipede/gnn/6 --test 100
```
The reward for **NerveNet** pretrained model is around: *2477*

To train an agent from sratch using NerveNet, you could use the following code:
```bash
python main.py --task ReacherOne-v1 --use_gnn_as_policy 1 --network_shape 64,64 --lr 0.0003 --num_threads 4 --lr_schedule adaptive --max_timesteps 1000000 --use_gnn_as_value 0 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB
```
