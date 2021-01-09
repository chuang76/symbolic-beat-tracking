# Beat and Downbeat Tracking of Symbolic Music Data

In this project, we construct a symbolic beat tracking system that performs joint beat and downbeat tracking in a multi-task learning (MTL) manner. The proposed models are based on variants of recurrent neural networks (RNN), All the models are implemented with the [PyTorch](https://pytorch.org/) framework. 

![](https://github.com/chuang76/symbolic-beat-tracking/blob/master/figure/test.png?raw=true)



## Network Structure

![](https://github.com/chuang76/symbolic-beat-tracking/blob/master/figure/network.png?raw=true)

In the network stage, we consider three types of neural networks: the first is the conventional bidirectional LSTM (BLSTM) network, the second is the Hierarchical Multiscale RNN (HM-RNN), and the third is the BLSTM with attention mechanism. For a more detailed discussion, please check our [paper](https://github.com/chuang76/symbolic-beat-tracking/blob/master/paper/Beat_Tracking_Symbolic_Music.pdf). 



## Environment

- Ubuntu 18.04
- Python 3.6.8

  

## Usage 

- **Get a copy**: Get a copy of this project by cloning the Git repository. 

  ```
  git clone https://github.com/chuang76/symbolic-beat-tracking.git
  ```

- **Prerequisite**: Install all the dependencies from requirements.txt. 

  ```
  pip install -r requirements.txt
  ```

- **Run**: 

  - **Input**: Put the symbolic music input files into `~/symbolic-beat-tracking/input/`. <br>We provide two kinds of input format: MIDI file or csv file. The input features include "start_time", "end_time", and "note" (i.e. pitch). If you choose to utilize csv file as input data, note that the start_time and end_time represent (onset time * sampling rate) and (offset time * sampling rate), respectively. You can acquire symbolic music data in csv format from the [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) dataset. 
  
    |      | **start_time** | **end_time** | **note** |
    | ---- | -------------- | ------------ | -------- |
    | 0    | 28126          | 29662        | 57       |
    | 1    | 46557          | 52702        | 73       |
    | 2    | 60893          | 109022       | 76       |
  
  - **Output:** Simply execute the make command, then you can obtain the beat and downbeat tracking results in `~/symbolic-beat-tracking/output/`. The tracking results contain the beat and downbeat positions in seconds. 
  
  ```
  make
  ```



## Example

Here is a short example. There are two symbolic music data files `01.csv` and `02.mid` in the input folder. You can check their beat/downbeat tracking results as follows. 

![](https://raw.githubusercontent.com/chuang76/symbolic-beat-tracking/master/figure/proc.png)



## Citation

Yi-Chin Chuang, Li Su, "Beat and Downbeat Tracking of Symbolic Music Data Using Deep Recurrent Neural Networks", Proceedings of Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), December 2020. 






