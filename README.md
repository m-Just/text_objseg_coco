usage  
1. go to direcotry refer  
`cd refer`  
2. download refcoco+ expression and annotation from [http://tlberg.cs.unc.edu/licheng/referit/data/refcoco+.zip] and extract `refcoco+` into `data`
2. extract `coco_bbox.zip`
3. download refcoco+ image dataset (this may take a long time)  
`python download_dataset.py`  
4. extract data from refcoco+  
`python parse_data.py`  
5. extract bounding box proposal  
`python parse_bbox.py`  
6. `cd ..` and train the model according to ronghang/text_objseg, change path `exp-referit` into `refer`

