# CS542_Project

Aiming to detect dangerous items on X-Ray.

Our project is tesed on BU SCC:
4 CPU cores and 1 GPU (12 GB memory) 
compute capability of at least 3.5 (which includes the K40m and P100 cards)).

## Requirements:
* python/3.6.2
* tensorflow/r1.10

## Installation:
* If you have account on BU SCC, just clone this repository. Open terminal under 
<a href = "/danger" title = "danger">danger</a></p>
  ><p>run qsub myJob.sh</p>
* Otherwise:
  >1. Clone this repository
  >2. Install dependencies
    >>* <p>pip3 install -r requirements.txt</p>
    >>* <p>python3 setup.py install</p>
  >3. Start training:
    >><p>python3 danger.py train --dataset=../dataset/danger --weights=coco</p>



