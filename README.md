usage  
1. go to direcotry refer  
`cd refer`  
2. download refcoco+ expressions and annotations from http://tlberg.cs.unc.edu/licheng/referit/data/refcoco+.zip  
3. `mkdir data` and extract `refcoco+.zip` into `data`  
4. extract `coco_bbox.zip`  
5. download refcoco+ image dataset (this may take a long time)  
`python download_dataset.py`  
6. extract data from refcoco+  
`python parse_data.py`  
7. extract bounding box proposal  
`python parse_bbox.py`  
8. `cd ..` and train the model according to ronghang/text_objseg, change path `exp-referit` into `refer`

