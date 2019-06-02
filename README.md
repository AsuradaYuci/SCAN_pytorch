# Something You should know
The official code for SCAN is [here](https://github.com/ruixuejianfei/SCAN).

The Optical Flow datasets for ilids and prid2011 are **not avaliable**, you should generate it by yourself,or you can choose don't use the Optical flow.

You can get the optical_flow data for **ilids_vid** from here https://drive.google.com/open?id=1u9jMd9wmmW25fAKGTRWg6pPRJtemcp4v, which is provided by https://github.com/dapengchen123/video_reid.

Don't forget **change** the optical_flow data **name** to make sure it's same as in **ilidsvidsequence.py** or **prid2011sequence.py**.

# What I do
1.fix some bugs to make code runable on

**pytorch>=0.4, Python>=3.6** 

2.the files that I have changed some where:

        a. seqtransforms.py
        b. sampler.py
        c. attevaluator.py
        d. classifer.py

# Self-and-Collaborative Attention Network

This solution contains source code of the project 
"SCAN: Self-and-Collaborative Attention Network for Video Person Re-identiﬁcation" 

The source code is for educational and research use only without any warranty; 
if you use any part of the source code, please cite related paper:


``` 
@article{zhang2018scan,
  title={SCAN: Self-and-Collaborative Attention Network for Video Person Re-identification},
  author={Zhang, Ruimao and Sun, Hongbin and Li, Jingyu and Ge, Yuying and Lin, Liang and Luo, Ping and Wang, Xiaogang},
  journal={IEEE Trans. on Image Processing},
  year={2019}
}
```


# License

All materials in this repository are released under the [CC-BY-NC 4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/).


