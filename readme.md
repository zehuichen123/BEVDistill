<div align="center">
<h1> BEVDistill </h1>
<h3>Cross-Modal BEV Distillation for Multi-View 3D Object Detection</h3>
<br>Zehui Chen, Zhenyu Li, Shiquan Zhang, Liangji Fang, Qinhong Jiang, Feng Zhao. 
<br>

<div><a href="https://arxiv.org/pdf/2211.09386.pdf">[Paper] </a></div> 

<center>
<img src='figs/framework.png'>
</center>

</div>

## Performance

### nuScenes Val set
| Model | config | mAP | NDS |
| - | - | - | - |
| BEVFormer-R50 | | 35.2 | 42.3 |
| BEVDistill-R50 | | 38.6 | 45.7 |
| BEVFormer-T-R50 | | 38.3 | 48.8 |
| BEVDistill-T-R50 | | 40.7 | 51.5 |

### nuScenes Test Leaderboard
| Model | mAP | NDS |
| -|-|-|
| BEVDistill |  49.8  |  59.4  |