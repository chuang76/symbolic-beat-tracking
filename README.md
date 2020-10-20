# Beat and Downbeat Tracking of Symbolic Music Data

In this project, we construct a symbolic beat tracking framework that performs joint beat and downbeat tracking in a multi-task learning (MTL) manner. The proposed models are based on recurrent neural networks (RNN), namely bidirectional long short-term memory (BLSTM) network, hierarchical multi-scale (HM) LSTM, and BLSTM with the attention mechanism.



## Network Structure

![](https://github.com/chuang76/symbolic-beat-tracking/blob/master/figure/network.PNG?raw=true)



## Usage 

- **Get a copy**: Get a copy of this project by running the git clone command. 

  ```
  $ git clone https://github.com/chuang76/symbolic-beat-tracking.git
  ```

- **Prerequisite**: Before running the project, you need to install all the dependencies from requirements.txt. 

  ```
  $ pip install -r requirements.txt
  ```

- **Execute**: 

  Put your input csv file into `~/symbolic-beat-tracking/input/`, and execute the make command. 
  
  Thus, you can check the beat tracking result in `~/symbolic-beat-tracking/output/`. The tracking result is a txt file, contains the beat positions in second. 
  
  ```
  $ make
  ```
  



## Example

There is an example file `test.csv` in the input folder. Let's check its beat tracking result with our proposed network.

 









