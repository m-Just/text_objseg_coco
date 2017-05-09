usage  
1. go to direcotry refer  
`cd refer`  
2. `mkdir coco_bbox` and put downloaded coco bounding box proposal in to the directory  
3. download refcoco+ image dataset (this may take a long time)  
`python download_dataset.py`  
4. extract data from refcoco+  
`python parse_data.py`  
5. extract bounding box proposal  
`python parse_bbox.py`  
6. `cd ..` and train the model according to ronghang/text_objseg, change path `exp-referit` into `refer`

