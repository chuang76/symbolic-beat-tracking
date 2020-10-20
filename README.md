# Beat and Downbeat Tracking of Symbolic Music Data

In this project, we construct a symbolic beat tracking framework that performs joint beat and downbeat tracking in a multi-task learning (MTL) manner. The proposed models are based on recurrent neural networks (RNN), namely bidirectional long short-term memory (BLSTM) network, hierarchical multi-scale (HM) LSTM, and BLSTM with the attention mechanism.



## Network Structure

![](https://github.com/chuang76/symbolic-beat-tracking/blob/master/figure/network.PNG?raw=true)



## Environment

- Ubuntu 18.04
- Python 3.6.8



## Usage 

- **Get a copy**: Get a copy of this project by running the git clone command. 

  ```
  $ git clone https://github.com/chuang76/symbolic-beat-tracking.git
  ```

- **Prerequisite**: Before running the project, you need to install all the dependencies from requirements.txt. 

  ```
  $ pip install -r requirements.txt
  ```

- **Execute**: Put your input csv file into `~/symbolic-beat-tracking/input/`, and execute the make command. Therefore, you can check the beat and downbeat tracking results in `~/symbolic-beat-tracking/output/`. The tracking results are text files, contains the beat and downbeat positions in second unit. 

  ```
  $ make
  ```



## Example

For example, there is an example file `test.csv` in the input folder. You can check its beat/downbeat tracking results with our proposed networks as follows. 

A simple visualization file `plot.py` is provided, you can modify it according to your needs. 









