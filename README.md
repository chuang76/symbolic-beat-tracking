# Beat and Downbeat Tracking of Symbolic Music Data

In this project, we construct a symbolic beat tracking framework that performs joint beat and downbeat tracking in a multi-task learning (MTL) manner. The proposed models are based on recurrent neural networks (RNN), namely bidirectional long short-term memory (BLSTM) network, hierarchical multi-scale (HM) LSTM, and BLSTM with the attention mechanism.

![](https://github.com/chuang76/symbolic-beat-tracking/blob/master/figure/test.png?raw=true)



## Network Structure

![](https://github.com/chuang76/symbolic-beat-tracking/blob/master/figure/network.png?raw=true)



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

- **Run**: 

  - Input: Put your symbolic music input files into `~/symbolic-beat-tracking/input/`. The input file should be in csv format, which stores 5 columns of note event information, namely "start_time", "end_time", "instrument", "note" (i.e. pitch), and "note_value". <br>Here, "start_time" and "end_time" represent (onset time * sampling rate) and (offset time * sampling rate), respectively. You can obtain symbolic music data directly from the [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) dataset or derive note event information with the [pretty_midi](https://craffel.github.io/pretty-midi/#pretty-midi-prettymidi) library. 
  
    |      | **start_time** | **end_time** | instrument | **note** | **note_value** |
    | ---- | -------------- | ------------ | ---------- | -------- | -------------- |
    | 0    | 28126          | 29662        | 43         | 57       | Eighth         |
    | 1    | 46557          | 52702        | 41         | 73       | Eighth         |
    | 2    | 60893          | 109022       | 41         | 76       | Dotted Half    |
  
  - Output: Execute the make command, then you can check the beat and downbeat tracking results in `~/symbolic-beat-tracking/output/`. The tracking results contain the beat and downbeat positions in second unit. 
  
    ```
    $ make
    ```



## Example

For example, there are two symbolic music data files `01.csv` and `02.csv` in the input folder. You can check their beat/downbeat tracking results with our proposed networks as follows. 

![](https://github.com/chuang76/symbolic-beat-tracking/blob/master/figure/proc.png?raw=true)


